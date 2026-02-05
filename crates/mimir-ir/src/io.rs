use std::io::{BufRead, Read, Write};

use postcard::{from_bytes, to_allocvec};

use crate::{ir::MimirIRData, util::error::ASTError};

impl MimirIRData {
    pub fn save(ir: &MimirIRData, write: &mut Box<dyn Write>) -> Result<(), ASTError> {
        let bytes: Vec<u8> = to_allocvec(ir)
            .map_err(|e| ASTError::Serialization(e.to_string()))?
            .to_vec();

        write.write_all(&bytes).map_err(ASTError::Io)?;

        Ok(())
    }

    pub fn load(read: &mut Box<dyn BufRead>) -> Result<MimirIRData, ASTError> {
        let mut bytes: Vec<u8> = vec![];

        let _ = read.read_to_end(&mut bytes).map_err(ASTError::Io)?;

        let ir_data: MimirIRData =
            from_bytes(&bytes).map_err(|e| ASTError::Deserialization(e.to_string()))?;

        Ok(ir_data)
    }
}
