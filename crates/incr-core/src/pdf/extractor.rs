//! PDF text and image extraction using lopdf and pdf-extract.

use image::{DynamicImage, ImageBuffer, Rgba};
use lopdf::{Document, Object, ObjectId};
use std::collections::HashSet;
use std::io::Cursor;
use tracing::{debug, trace};

use super::{PdfProcessor, PdfType, Result};
use crate::error::PdfError;

/// PDF content extractor using lopdf.
pub struct PdfExtractor {
    document: Option<Document>,
    raw_data: Vec<u8>,
}

/// Extracted content from a PDF.
#[derive(Debug, Clone)]
pub struct PdfContent {
    /// Type of PDF content.
    pub pdf_type: PdfType,
    /// Extracted text (if any).
    pub text: String,
    /// Pages with their content.
    pub pages: Vec<PdfPage>,
}

/// Content from a single PDF page.
#[derive(Debug, Clone)]
pub struct PdfPage {
    /// Page number (1-indexed).
    pub number: u32,
    /// Extracted text from this page.
    pub text: String,
    /// Images extracted from this page.
    pub images: Vec<ExtractedImage>,
}

/// An image extracted from a PDF.
#[derive(Debug, Clone)]
pub struct ExtractedImage {
    /// Image data.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Image format (jpeg, png, etc.).
    pub format: String,
}

impl PdfExtractor {
    /// Create a new PDF extractor.
    pub fn new() -> Self {
        Self {
            document: None,
            raw_data: Vec::new(),
        }
    }

    /// Load and extract all content from a PDF.
    pub fn extract_all(&self) -> Result<PdfContent> {
        let doc = self.document.as_ref().ok_or(PdfError::Parse("No document loaded".to_string()))?;

        let page_count = doc.get_pages().len() as u32;
        if page_count == 0 {
            return Err(PdfError::NoPages);
        }

        let mut pages = Vec::with_capacity(page_count as usize);
        let mut full_text = String::new();
        let mut total_text_len = 0;
        let mut total_images = 0;

        for page_num in 1..=page_count {
            let page_text = self.extract_page_text(page_num).unwrap_or_default();
            let images = self.extract_images(page_num).unwrap_or_default();

            total_text_len += page_text.len();
            total_images += images.len();

            if !page_text.is_empty() {
                if !full_text.is_empty() {
                    full_text.push_str("\n\n");
                }
                full_text.push_str(&page_text);
            }

            let extracted_images: Vec<ExtractedImage> = images
                .into_iter()
                .map(|img| {
                    let (width, height) = (img.width(), img.height());
                    let mut data = Vec::new();
                    img.write_to(&mut Cursor::new(&mut data), image::ImageFormat::Png)
                        .unwrap_or_default();
                    ExtractedImage {
                        data,
                        width,
                        height,
                        format: "png".to_string(),
                    }
                })
                .collect();

            pages.push(PdfPage {
                number: page_num,
                text: page_text,
                images: extracted_images,
            });
        }

        let pdf_type = if total_text_len > 50 && total_images == 0 {
            PdfType::Text
        } else if total_text_len <= 50 && total_images > 0 {
            PdfType::Image
        } else if total_text_len > 50 && total_images > 0 {
            PdfType::Hybrid
        } else {
            PdfType::Empty
        };

        debug!(
            "PDF analysis: {} pages, {} chars text, {} images -> {:?}",
            page_count, total_text_len, total_images, pdf_type
        );

        Ok(PdfContent {
            pdf_type,
            text: full_text,
            pages,
        })
    }

    /// Extract all images from the entire document
    fn extract_all_images(&self) -> Vec<DynamicImage> {
        let doc = match self.document.as_ref() {
            Some(d) => d,
            None => return vec![],
        };

        let mut images = Vec::new();
        let mut seen_objects: HashSet<ObjectId> = HashSet::new();

        // Iterate through all objects in the document
        for (id, object) in doc.objects.iter() {
            if seen_objects.contains(id) {
                continue;
            }
            seen_objects.insert(*id);

            if let Some(img) = self.try_extract_image_from_object(doc, object) {
                images.push(img);
            }
        }

        debug!("Found {} images in document", images.len());
        images
    }

    fn try_extract_image_from_object(&self, doc: &Document, obj: &Object) -> Option<DynamicImage> {
        if let Object::Stream(stream) = obj {
            let dict = &stream.dict;

            // Check if it's an image XObject
            let subtype = dict.get(b"Subtype").ok()?;
            if subtype.as_name().ok()? != b"Image" {
                return None;
            }

            let width = dict.get(b"Width").ok()?.as_i64().ok()? as u32;
            let height = dict.get(b"Height").ok()?.as_i64().ok()? as u32;

            trace!("Found image object: {}x{}", width, height);

            // Get the decompressed stream content
            let data = match stream.decompressed_content() {
                Ok(d) => d,
                Err(_) => stream.content.clone(),
            };

            // Check for image filters
            if let Ok(filter) = dict.get(b"Filter") {
                let filter_name = match filter {
                    Object::Name(name) => Some(name.as_slice()),
                    Object::Array(arr) if !arr.is_empty() => {
                        arr.first().and_then(|o| o.as_name().ok())
                    }
                    _ => None,
                };

                match filter_name {
                    Some(b"DCTDecode") => {
                        // JPEG data - use raw stream content (already compressed)
                        trace!("Decoding JPEG image");
                        return image::load_from_memory_with_format(&stream.content, image::ImageFormat::Jpeg).ok();
                    }
                    Some(b"JPXDecode") => {
                        // JPEG 2000
                        trace!("Found JPEG2000 image (not supported)");
                        return None;
                    }
                    Some(b"CCITTFaxDecode") | Some(b"JBIG2Decode") => {
                        // Fax/JBIG2 - complex to decode
                        trace!("Found fax/JBIG2 image (not supported)");
                        return None;
                    }
                    _ => {}
                }
            }

            // Try to decode raw image data
            let color_space = dict
                .get(b"ColorSpace")
                .ok()
                .and_then(|o| match o {
                    Object::Name(name) => Some(name.as_slice()),
                    Object::Array(arr) => arr.first().and_then(|o| o.as_name().ok()),
                    Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| o.as_name().ok()),
                    _ => None,
                })
                .unwrap_or(b"DeviceRGB");

            let bits = dict
                .get(b"BitsPerComponent")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .unwrap_or(8) as u8;

            return self.create_image_from_raw(&data, width, height, color_space, bits);
        }
        None
    }

    fn create_image_from_raw(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        color_space: &[u8],
        bits_per_component: u8,
    ) -> Option<DynamicImage> {
        trace!(
            "Creating image from raw data: {}x{}, colorspace={:?}, bits={}",
            width, height, String::from_utf8_lossy(color_space), bits_per_component
        );

        if bits_per_component != 8 {
            trace!("Unsupported bits per component: {}", bits_per_component);
            return None;
        }

        let expected_rgb = (width * height * 3) as usize;
        let expected_gray = (width * height) as usize;

        if color_space == b"DeviceRGB" || color_space == b"RGB" {
            if data.len() >= expected_rgb {
                let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
                for chunk in data[..expected_rgb].chunks(3) {
                    if chunk.len() == 3 {
                        rgba_data.push(chunk[0]);
                        rgba_data.push(chunk[1]);
                        rgba_data.push(chunk[2]);
                        rgba_data.push(255);
                    }
                }
                return ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba_data)
                    .map(DynamicImage::ImageRgba8);
            }
        } else if color_space == b"DeviceGray" || color_space == b"G" {
            if data.len() >= expected_gray {
                let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
                for &gray in data[..expected_gray].iter() {
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(255);
                }
                return ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba_data)
                    .map(DynamicImage::ImageRgba8);
            }
        }

        trace!("Could not decode image: data_len={}, expected_rgb={}, expected_gray={}",
               data.len(), expected_rgb, expected_gray);
        None
    }

    /// Get resources dictionary for a page, handling inheritance
    fn get_page_resources(&self, doc: &Document, page_id: ObjectId) -> Option<lopdf::Dictionary> {
        let page = doc.get_object(page_id).ok()?;
        if let Object::Dictionary(dict) = page {
            // First check if Resources is directly on the page
            if let Ok(resources) = dict.get(b"Resources") {
                if let Ok((_, Object::Dictionary(res_dict))) = doc.dereference(resources) {
                    return Some(res_dict.clone());
                }
            }

            // Check parent for inherited Resources
            if let Ok(parent_ref) = dict.get(b"Parent") {
                if let Object::Reference(parent_id) = parent_ref {
                    return self.get_inherited_resources(doc, *parent_id);
                }
            }
        }
        None
    }

    fn get_inherited_resources(&self, doc: &Document, node_id: ObjectId) -> Option<lopdf::Dictionary> {
        let node = doc.get_object(node_id).ok()?;
        if let Object::Dictionary(dict) = node {
            // Check for Resources
            if let Ok(resources) = dict.get(b"Resources") {
                if let Ok((_, Object::Dictionary(res_dict))) = doc.dereference(resources) {
                    return Some(res_dict.clone());
                }
            }

            // Continue up the tree
            if let Ok(parent_ref) = dict.get(b"Parent") {
                if let Object::Reference(parent_id) = parent_ref {
                    return self.get_inherited_resources(doc, *parent_id);
                }
            }
        }
        None
    }
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PdfProcessor for PdfExtractor {
    fn load(&mut self, data: &[u8]) -> Result<()> {
        let mut doc = Document::load_mem(data).map_err(|e| PdfError::Parse(e.to_string()))?;

        // Handle PDFs with empty password encryption
        if doc.is_encrypted() {
            // Try to decrypt with empty password
            if doc.decrypt("").is_err() {
                return Err(PdfError::Encrypted);
            }
            debug!("Decrypted PDF with empty password");

            // Save decrypted document to raw_data for pdf_extract
            let mut decrypted_data = Vec::new();
            doc.save_to(&mut decrypted_data)
                .map_err(|e| PdfError::Parse(format!("Failed to save decrypted PDF: {}", e)))?;
            self.raw_data = decrypted_data;
        } else {
            self.raw_data = data.to_vec();
        }

        let page_count = doc.get_pages().len();
        if page_count == 0 {
            return Err(PdfError::NoPages);
        }

        debug!("Loaded PDF with {} pages", page_count);
        self.document = Some(doc);
        Ok(())
    }

    fn page_count(&self) -> u32 {
        self.document
            .as_ref()
            .map(|doc| doc.get_pages().len() as u32)
            .unwrap_or(0)
    }

    fn analyze(&self) -> PdfType {
        let text = self.extract_text().unwrap_or_default();
        let has_text = text.len() > 50;

        // Check for images by scanning all objects
        let images = self.extract_all_images();
        let has_images = !images.is_empty();

        let pdf_type = match (has_text, has_images) {
            (true, false) => PdfType::Text,
            (false, true) => PdfType::Image,
            (true, true) => PdfType::Hybrid,
            (false, false) => PdfType::Empty,
        };

        debug!("PDF analysis: has_text={}, has_images={} -> {:?}", has_text, has_images, pdf_type);
        pdf_type
    }

    fn extract_text(&self) -> Result<String> {
        let text = pdf_extract::extract_text_from_mem(&self.raw_data)
            .map_err(|e| PdfError::TextExtraction(e.to_string()))?;
        Ok(text)
    }

    fn extract_page_text(&self, page: u32) -> Result<String> {
        // Use full text extraction and try to get the page portion
        let full_text = self.extract_text()?;
        let lines: Vec<&str> = full_text.lines().collect();
        let page_count = self.page_count() as usize;

        if page_count == 0 {
            return Ok(String::new());
        }

        let lines_per_page = lines.len() / page_count;
        let start = ((page - 1) as usize) * lines_per_page;
        let end = (page as usize) * lines_per_page;

        Ok(lines[start.min(lines.len())..end.min(lines.len())].join("\n"))
    }

    fn render_page(&self, page: u32, _dpi: u32) -> Result<DynamicImage> {
        // Try to extract images from the page
        let images = self.extract_images(page)?;

        if let Some(first) = images.into_iter().next() {
            return Ok(first);
        }

        // Fall back to extracting all images from document
        let all_images = self.extract_all_images();
        let page_idx = (page - 1) as usize;

        if page_idx < all_images.len() {
            return Ok(all_images.into_iter().nth(page_idx).unwrap());
        }

        if let Some(first) = all_images.into_iter().next() {
            return Ok(first);
        }

        Err(PdfError::ImageExtraction(
            "No images found in PDF".to_string(),
        ))
    }

    fn extract_images(&self, page: u32) -> Result<Vec<DynamicImage>> {
        let doc = self.document.as_ref().ok_or(PdfError::Parse("No document loaded".to_string()))?;

        let pages = doc.get_pages();
        let page_id = pages
            .get(&page)
            .ok_or(PdfError::InvalidPage(page))?;

        let mut images = Vec::new();

        // Get resources (with inheritance support)
        if let Some(resources) = self.get_page_resources(doc, *page_id) {
            // Look for XObjects
            if let Ok(xobjects) = resources.get(b"XObject") {
                if let Ok((_, Object::Dictionary(xobj_dict))) = doc.dereference(xobjects) {
                    for (_name, obj_ref) in xobj_dict.iter() {
                        if let Ok((_, obj)) = doc.dereference(obj_ref) {
                            if let Some(img) = self.try_extract_image_from_object(doc, obj) {
                                images.push(img);
                            }
                        }
                    }
                }
            }
        }

        // If no images found via XObject, try scanning all objects
        if images.is_empty() {
            debug!("No XObject images found on page {}, scanning all objects", page);
            images = self.extract_all_images();
        }

        debug!("Extracted {} images from page {}", images.len(), page);
        Ok(images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_extractor_new() {
        let extractor = PdfExtractor::new();
        assert!(extractor.document.is_none());
        assert_eq!(extractor.page_count(), 0);
    }
}
