//! Embedded model data for standalone binary distribution.
//!
//! Mobile models are embedded directly in the binary for easy distribution.
//! Server models must be downloaded separately due to size.

/// Embedded mobile detection model (~4.5MB)
pub static MOBILE_DET: &[u8] = include_bytes!("../../../../models/mobile/det.onnx");

/// Embedded mobile recognition model (~7.5MB)
pub static MOBILE_REC: &[u8] = include_bytes!("../../../../models/mobile/latin_rec.onnx");

/// Embedded mobile layout model (~7.1MB)
pub static MOBILE_LAYOUT: &[u8] = include_bytes!("../../../../models/mobile/layout.onnx");

/// Embedded mobile dictionary (~1.6KB)
pub static MOBILE_DICT: &str = include_str!("../../../../models/mobile/latin_dict.txt");

/// Check if embedded models are available.
pub fn has_embedded_models() -> bool {
    // Always true when compiled with embedded models
    !MOBILE_DET.is_empty()
}

/// Get embedded model data for mobile variant.
pub struct EmbeddedModels {
    pub detection: &'static [u8],
    pub recognition: &'static [u8],
    pub layout: &'static [u8],
    pub dictionary: &'static str,
}

impl EmbeddedModels {
    /// Get mobile embedded models.
    pub fn mobile() -> Self {
        Self {
            detection: MOBILE_DET,
            recognition: MOBILE_REC,
            layout: MOBILE_LAYOUT,
            dictionary: MOBILE_DICT,
        }
    }
}
