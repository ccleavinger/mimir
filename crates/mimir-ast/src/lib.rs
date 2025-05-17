#[cfg(feature = "bin")]
use std::fs::File;

#[cfg(not(feature = "bin"))]
use std::{fs::File, io::Read};

use serde::{Deserialize, Serialize};
use syn::{FnArg, ItemFn, Pat};
use quote::ToTokens;

#[cfg(feature = "bin")]
use bincode;

// Represents a function parameter
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_name: String,
    pub is_mutable: bool,
}

// Main AST structure that will be serialized
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MimirGlobalAST {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: String,
}

impl MimirGlobalAST {
    pub fn from_syn_fn(func: &ItemFn) -> Self {
        // Extract parameters
        let parameters = func.sig.inputs.iter().map(|arg| {
            match arg {
                FnArg::Typed(pat_type) => {
                    let name = match &*pat_type.pat {
                        Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                        _ => "unknown".to_string(),
                    };
                    
                    let type_name = pat_type.ty.to_token_stream().to_string();
                    let is_mutable = match &*pat_type.pat {
                        Pat::Ident(pat_ident) => pat_ident.mutability.is_some(),
                        _ => false,
                    };
                    
                    Parameter {
                        name,
                        type_name,
                        is_mutable,
                    }
                },
                // this is undesired behavior. TODO: gracefully handle this edge case
                FnArg::Receiver(_) => Parameter {
                    name: "self".to_string(),
                    type_name: "Self".to_string(),
                    is_mutable: false,
                },
            }
        }).collect();

        MimirGlobalAST {
            name: func.sig.ident.to_string(),
            parameters,
            body: func.block.to_token_stream().to_string(),
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(feature = "json")] {
            pub fn save_to_file(asts: &Vec<Self>, path: &str) -> std::io::Result<()> {
                let file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
                serde_json::to_writer_pretty(file, asts)?;
                Ok(())
            }

            pub fn load_from_file(path: &str) -> std::io::Result<Vec<Self>> {
                let mut file = File::open(path)?;
                let mut content = String::new();
                file.read_to_string(&mut content)?;
                let asts: Vec<Self> = serde_json::from_str(&content)?;
                Ok(asts)
            }
        } else {
            pub fn save_to_file(asts: &Vec<Self>, path: &str) -> std::io::Result<()> {
                let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
        
                bincode::serde::encode_into_std_write(asts, &mut file, bincode::config::standard())
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
                Ok(())
            }

            pub fn load_from_file(path: &str) -> std::io::Result<Vec<Self>> {
                let mut file = File::open(path)?;
                let asts: Vec<Self> = bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                Ok(asts)
            }
        }
    }
}

pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct Usize3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}