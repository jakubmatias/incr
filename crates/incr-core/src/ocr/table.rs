//! Table structure recognition using SLANet model.
//!
//! Extracts table structure (rows, columns, cells) from table images.

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;
use tracing::debug;

use crate::error::OcrError;
use incr_inference::{InferenceBackend, InputTensor, OutputTensor};

/// A cell in a table.
#[derive(Debug, Clone)]
pub struct TableCell {
    /// Row index (0-based).
    pub row: usize,
    /// Column index (0-based).
    pub col: usize,
    /// Row span (number of rows this cell spans).
    pub row_span: usize,
    /// Column span (number of columns this cell spans).
    pub col_span: usize,
    /// Bounding box in image coordinates (x1, y1, x2, y2).
    pub bbox: [f32; 4],
    /// Cell content (text, filled by OCR separately).
    pub content: String,
    /// Confidence score.
    pub confidence: f32,
}

impl TableCell {
    /// Check if this cell spans multiple rows.
    pub fn is_row_spanning(&self) -> bool {
        self.row_span > 1
    }

    /// Check if this cell spans multiple columns.
    pub fn is_col_spanning(&self) -> bool {
        self.col_span > 1
    }

    /// Get the area of the cell.
    pub fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    }
}

/// A recognized table structure.
#[derive(Debug, Clone)]
pub struct TableStructure {
    /// Number of rows.
    pub num_rows: usize,
    /// Number of columns.
    pub num_cols: usize,
    /// All cells in the table.
    pub cells: Vec<TableCell>,
    /// HTML representation of the table structure.
    pub html: String,
    /// Bounding box of the entire table.
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
}

impl TableStructure {
    /// Get cells in a specific row.
    pub fn row(&self, row: usize) -> Vec<&TableCell> {
        self.cells.iter().filter(|c| c.row == row).collect()
    }

    /// Get cells in a specific column.
    pub fn column(&self, col: usize) -> Vec<&TableCell> {
        self.cells.iter().filter(|c| c.col == col).collect()
    }

    /// Get the cell at a specific position.
    pub fn cell_at(&self, row: usize, col: usize) -> Option<&TableCell> {
        self.cells.iter().find(|c| {
            row >= c.row
                && row < c.row + c.row_span
                && col >= c.col
                && col < c.col + c.col_span
        })
    }

    /// Get header row (first row).
    pub fn header(&self) -> Vec<&TableCell> {
        self.row(0)
    }

    /// Get data rows (all rows except header).
    pub fn data_rows(&self) -> Vec<Vec<&TableCell>> {
        (1..self.num_rows).map(|r| self.row(r)).collect()
    }

    /// Convert to a 2D grid of cell references.
    pub fn as_grid(&self) -> Vec<Vec<Option<&TableCell>>> {
        let mut grid = vec![vec![None; self.num_cols]; self.num_rows];

        for cell in &self.cells {
            for r in cell.row..(cell.row + cell.row_span).min(self.num_rows) {
                for c in cell.col..(cell.col + cell.col_span).min(self.num_cols) {
                    grid[r][c] = Some(cell);
                }
            }
        }

        grid
    }

    /// Generate HTML from table structure.
    pub fn to_html(&self) -> String {
        let mut html = String::from("<table>\n");

        for row_idx in 0..self.num_rows {
            html.push_str("  <tr>\n");

            let mut col_idx = 0;
            while col_idx < self.num_cols {
                if let Some(cell) = self.cells.iter().find(|c| c.row == row_idx && c.col == col_idx)
                {
                    let tag = if row_idx == 0 { "th" } else { "td" };
                    let mut attrs = String::new();

                    if cell.row_span > 1 {
                        attrs.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
                    }
                    if cell.col_span > 1 {
                        attrs.push_str(&format!(" colspan=\"{}\"", cell.col_span));
                    }

                    html.push_str(&format!(
                        "    <{}{}>{}</{}>\n",
                        tag, attrs, cell.content, tag
                    ));

                    col_idx += cell.col_span;
                } else {
                    // Cell is covered by a spanning cell from above
                    col_idx += 1;
                }
            }

            html.push_str("  </tr>\n");
        }

        html.push_str("</table>");
        html
    }
}

/// Table type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableType {
    /// Table with visible grid lines.
    Wired,
    /// Table without visible grid lines.
    Lineless,
    /// Unknown table type.
    Unknown,
}

/// Table structure recognizer using SLANet model.
pub struct TableRecognizer<B: InferenceBackend> {
    backend: B,
    input_size: (u32, u32),
    max_length: usize,
}

impl<B: InferenceBackend> TableRecognizer<B> {
    /// Create a new table recognizer.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            input_size: (488, 488), // SLANet default
            max_length: 500,
        }
    }

    /// Set input size.
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = (width, height);
        self
    }

    /// Set maximum sequence length.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Recognize table structure from an image.
    pub fn recognize(&self, image: &DynamicImage) -> Result<TableStructure, OcrError> {
        let (orig_width, orig_height) = image.dimensions();

        // Preprocess
        let (tensor, scale_x, scale_y) = self.preprocess(image)?;

        debug!(
            "Table recognition input: {}x{}, scales: ({:.3}, {:.3})",
            self.input_size.0, self.input_size.1, scale_x, scale_y
        );

        // Run inference
        let input = InputTensor::Float32(tensor.into_dyn());
        let outputs = self
            .backend
            .run(&[("x", input)])
            .map_err(|e| OcrError::Detection(format!("Table recognition failed: {}", e)))?;

        // Parse outputs
        let structure =
            self.post_process(&outputs, scale_x, scale_y, orig_width, orig_height)?;

        debug!(
            "Recognized table: {}x{} with {} cells",
            structure.num_rows, structure.num_cols, structure.cells.len()
        );

        Ok(structure)
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<(Array3<f32>, f32, f32), OcrError> {
        let (orig_w, orig_h) = image.dimensions();
        let (target_w, target_h) = self.input_size;

        // Resize with padding to maintain aspect ratio
        let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        let resized = image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);

        let scale_x = new_w as f32 / orig_w as f32;
        let scale_y = new_h as f32 / orig_h as f32;

        // Create padded image
        let mut padded =
            image::RgbImage::from_pixel(target_w, target_h, image::Rgb([255, 255, 255]));

        // Calculate padding
        let pad_x = (target_w - new_w) / 2;
        let pad_y = (target_h - new_h) / 2;

        // Copy resized image to center
        let rgb = resized.to_rgb8();
        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = rgb.get_pixel(x, y);
                padded.put_pixel(x + pad_x, y + pad_y, *pixel);
            }
        }

        // Convert to CHW format with normalization
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let mut tensor = Array3::<f32>::zeros((3, target_h as usize, target_w as usize));

        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = padded.get_pixel(x as u32, y as u32);
                tensor[[0, y, x]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
                tensor[[1, y, x]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
                tensor[[2, y, x]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
            }
        }

        Ok((tensor, scale_x, scale_y))
    }

    fn post_process(
        &self,
        outputs: &[(String, OutputTensor)],
        scale_x: f32,
        scale_y: f32,
        orig_width: u32,
        orig_height: u32,
    ) -> Result<TableStructure, OcrError> {
        // SLANet outputs structure tokens and bounding boxes
        // Tokens represent HTML-like structure: <tr>, </tr>, <td>, </td>, <td rowspan="X">, etc.

        // Find structure and bbox outputs
        let structure_output = outputs
            .iter()
            .find(|(name, _)| name.contains("structure") || name.contains("output"))
            .or_else(|| outputs.first());

        let bbox_output = outputs
            .iter()
            .find(|(name, _)| name.contains("bbox") || name.contains("loc"));

        // Parse structure tokens
        let (cells, num_rows, num_cols) = if let Some((_, tensor)) = structure_output {
            self.parse_structure_tokens(tensor, bbox_output.map(|(_, t)| t), scale_x, scale_y)?
        } else {
            // Fallback: create a simple single-cell structure
            (
                vec![TableCell {
                    row: 0,
                    col: 0,
                    row_span: 1,
                    col_span: 1,
                    bbox: [0.0, 0.0, orig_width as f32, orig_height as f32],
                    content: String::new(),
                    confidence: 1.0,
                }],
                1,
                1,
            )
        };

        let html = self.build_html(&cells, num_rows, num_cols);

        Ok(TableStructure {
            num_rows,
            num_cols,
            cells,
            html,
            bbox: [0.0, 0.0, orig_width as f32, orig_height as f32],
            confidence: 1.0,
        })
    }

    fn parse_structure_tokens(
        &self,
        structure: &OutputTensor,
        bboxes: Option<&OutputTensor>,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<(Vec<TableCell>, usize, usize), OcrError> {
        // SLANet structure output is typically token indices
        // We need to decode these into cell information

        let structure_arr = match structure {
            OutputTensor::Float32(arr) => arr,
            OutputTensor::Int64(arr) => {
                // Convert int64 to process as tokens
                let tokens: Vec<i64> = arr.iter().copied().collect();
                return self.decode_tokens(&tokens, bboxes, scale_x, scale_y);
            }
            _ => {
                return Err(OcrError::Detection(
                    "Unexpected structure output type".to_string(),
                ))
            }
        };

        // If float output, find argmax for each position
        let shape = structure_arr.shape();
        if shape.len() < 2 {
            return Err(OcrError::Detection("Invalid structure shape".to_string()));
        }

        let seq_len = shape[shape.len() - 2];
        let vocab_size = shape[shape.len() - 1];

        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;

            for j in 0..vocab_size {
                let val = if shape.len() == 3 {
                    structure_arr[[0, i, j]]
                } else {
                    structure_arr[[i, j]]
                };

                if val > max_val {
                    max_val = val;
                    max_idx = j as i64;
                }
            }
            tokens.push(max_idx);
        }

        self.decode_tokens(&tokens, bboxes, scale_x, scale_y)
    }

    fn decode_tokens(
        &self,
        tokens: &[i64],
        bboxes: Option<&OutputTensor>,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<(Vec<TableCell>, usize, usize), OcrError> {
        // SLANet token vocabulary (simplified):
        // 0: <pad>
        // 1: <sos>
        // 2: <eos>
        // 3: <td>
        // 4: </td>
        // 5: <tr>
        // 6: </tr>
        // 7+: <td colspan="N">, <td rowspan="N">, etc.

        let bbox_data = bboxes.and_then(|t| match t {
            OutputTensor::Float32(arr) => Some(arr),
            _ => None,
        });

        let mut cells = Vec::new();
        let mut current_row = 0;
        let mut current_col = 0;
        let mut max_cols = 0;
        let mut cell_idx = 0;

        let mut in_cell = false;
        let mut cell_row_span = 1;
        let mut cell_col_span = 1;

        for &token in tokens {
            match token {
                2 => break,           // <eos>
                5 => {
                    // <tr>
                    current_col = 0;
                }
                6 => {
                    // </tr>
                    max_cols = max_cols.max(current_col);
                    current_row += 1;
                }
                3 => {
                    // <td>
                    in_cell = true;
                    cell_row_span = 1;
                    cell_col_span = 1;
                }
                4 => {
                    // </td>
                    if in_cell {
                        // Get bbox for this cell
                        let bbox = if let Some(arr) = bbox_data {
                            let shape = arr.shape();
                            if shape.len() >= 2 && cell_idx < shape[shape.len() - 2] {
                                let base = if shape.len() == 3 { 0 } else { 0 };
                                [
                                    arr[[base, cell_idx, 0]] / scale_x,
                                    arr[[base, cell_idx, 1]] / scale_y,
                                    arr[[base, cell_idx, 2]] / scale_x,
                                    arr[[base, cell_idx, 3]] / scale_y,
                                ]
                            } else {
                                [0.0, 0.0, 0.0, 0.0]
                            }
                        } else {
                            [0.0, 0.0, 0.0, 0.0]
                        };

                        cells.push(TableCell {
                            row: current_row,
                            col: current_col,
                            row_span: cell_row_span,
                            col_span: cell_col_span,
                            bbox,
                            content: String::new(),
                            confidence: 1.0,
                        });

                        current_col += cell_col_span;
                        cell_idx += 1;
                        in_cell = false;
                    }
                }
                t if t >= 7 && t < 20 => {
                    // colspan tokens (simplified mapping)
                    cell_col_span = ((t - 7) as usize).max(1).min(10);
                }
                t if t >= 20 && t < 33 => {
                    // rowspan tokens (simplified mapping)
                    cell_row_span = ((t - 20) as usize).max(1).min(10);
                }
                _ => {}
            }
        }

        let num_rows = current_row.max(1);
        let num_cols = max_cols.max(1);

        Ok((cells, num_rows, num_cols))
    }

    fn build_html(&self, cells: &[TableCell], num_rows: usize, num_cols: usize) -> String {
        let mut html = String::from("<table>\n");

        for row_idx in 0..num_rows {
            html.push_str("  <tr>\n");

            let mut col_idx = 0;
            while col_idx < num_cols {
                if let Some(cell) = cells.iter().find(|c| c.row == row_idx && c.col == col_idx) {
                    let tag = if row_idx == 0 { "th" } else { "td" };
                    let mut attrs = String::new();

                    if cell.row_span > 1 {
                        attrs.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
                    }
                    if cell.col_span > 1 {
                        attrs.push_str(&format!(" colspan=\"{}\"", cell.col_span));
                    }

                    html.push_str(&format!(
                        "    <{}{}>{}</{}>\n",
                        tag, attrs, cell.content, tag
                    ));

                    col_idx += cell.col_span;
                } else {
                    col_idx += 1;
                }
            }

            html.push_str("  </tr>\n");
        }

        html.push_str("</table>");
        html
    }
}

/// Table type classifier.
pub struct TableClassifier<B: InferenceBackend> {
    backend: B,
    input_size: (u32, u32),
}

impl<B: InferenceBackend> TableClassifier<B> {
    /// Create a new table classifier.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            input_size: (224, 224),
        }
    }

    /// Classify table type (wired vs lineless).
    pub fn classify(&self, image: &DynamicImage) -> Result<(TableType, f32), OcrError> {
        let (target_w, target_h) = self.input_size;

        // Preprocess
        let resized = image.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);

        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let rgb = resized.to_rgb8();
        let mut tensor = Array3::<f32>::zeros((3, target_h as usize, target_w as usize));

        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                tensor[[0, y, x]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
                tensor[[1, y, x]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
                tensor[[2, y, x]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
            }
        }

        // Run inference
        let input = InputTensor::Float32(tensor.into_dyn());
        let outputs = self
            .backend
            .run(&[("x", input)])
            .map_err(|e| OcrError::Detection(format!("Table classification failed: {}", e)))?;

        // Parse output
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::Detection("No output from classifier".to_string()))?
            .1;

        let arr = match output {
            OutputTensor::Float32(arr) => arr,
            _ => {
                return Err(OcrError::Detection(
                    "Unexpected output type".to_string(),
                ))
            }
        };

        // Binary classification: [wired_score, lineless_score]
        let wired_score = arr.get([0, 0]).copied().unwrap_or(0.0);
        let lineless_score = arr.get([0, 1]).copied().unwrap_or(0.0);

        // Apply softmax
        let max_score = wired_score.max(lineless_score);
        let wired_exp = (wired_score - max_score).exp();
        let lineless_exp = (lineless_score - max_score).exp();
        let sum = wired_exp + lineless_exp;

        let wired_prob = wired_exp / sum;
        let lineless_prob = lineless_exp / sum;

        if wired_prob > lineless_prob {
            Ok((TableType::Wired, wired_prob))
        } else {
            Ok((TableType::Lineless, lineless_prob))
        }
    }
}
