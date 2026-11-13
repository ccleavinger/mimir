//! Versioned transport format, separate from canonical identity bytes.
use super::Module;
use thiserror::Error;

pub const ARTIFACT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"MIMIRIR\0";

#[derive(Debug, Clone, Error)]
pub enum ArtifactError {
    #[error("invalid or truncated Mimir IR header")]
    Header,
    #[error("unsupported Mimir IR version {0}")]
    Version(u32),
    #[error("invalid Mimir IR encoding: {0}")]
    Codec(#[from] postcard::Error),
    #[error("trailing bytes after Mimir IR module")]
    TrailingBytes,
}

impl Module {
    pub fn to_bytes(&self) -> Result<Vec<u8>, ArtifactError> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
        bytes.extend(postcard::to_allocvec(self)?);
        Ok(bytes)
    }

    /// Decoding does not establish validity; call `validate` before use.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(ArtifactError::Header);
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != ARTIFACT_VERSION {
            return Err(ArtifactError::Version(version));
        }
        let (module, rest) = postcard::take_from_bytes(&bytes[12..])?;
        if !rest.is_empty() {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(module)
    }
}
