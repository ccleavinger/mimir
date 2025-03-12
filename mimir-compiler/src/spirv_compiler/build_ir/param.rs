use crate::spirv_compiler::compiler::SpirVCompiler;
use crate::spirv_compiler::ir::{MimirPtrType, MimirType, MimirVariable};
use rspirv::spirv::{StorageClass, Word};

impl SpirVCompiler {
    pub fn params_to_spirv(&mut self) -> Result<(), String> {
        self.buffers_to_spirv()?;

        self.push_const_to_spirv()?;

        Ok(())
    }

    fn buffers_to_spirv(&mut self) -> Result<(), String> {
        for (str, ty) in self
            .vars
            .iter()
            .filter(|(_, var)| var.ty.storage_class == StorageClass::Uniform)
            .map(|(str, var)| (str.clone(), var.ty.clone()))
            .collect::<Vec<(String, MimirPtrType)>>()
        {
            if let MimirType::RuntimeArray(arr_ty) = &ty.clone().base.clone() {
                let base_ty = ty.clone().base;

                let ptr_ty = if !self.types.contains_key(&base_ty)
                    || !self.ptr_types.contains_key(&ty.clone())
                {
                    let base_word = self.types.get(&arr_ty).ok_or(format!(
                        "failed to retrieve base type of runtime array: {}",
                        arr_ty
                    ))?;

                    let runtime_ty = self.spirv_builder.type_runtime_array(*base_word);

                    let struct_ty = self.spirv_builder.type_struct(vec![runtime_ty]);

                    self.types.insert(
                        MimirType::RuntimeArray(Box::new(base_ty.clone())),
                        struct_ty,
                    );

                    let ptr_ty = self
                        .spirv_builder
                        .type_pointer(None, ty.storage_class, struct_ty);

                    self.ptr_types.insert(ty.clone(), ptr_ty);

                    ptr_ty
                } else {
                    self.ptr_types.get(&ty).unwrap().clone()
                };

                let word = self
                    .spirv_builder
                    .variable(ptr_ty, None, ty.storage_class, None);

                self.spirv_builder.name(word, &str);

                self.vars.insert(
                    str,
                    MimirVariable {
                        ty: ty.clone(),
                        word: Some(word),
                    },
                );
            }
        }

        Ok(())
    }

    fn push_const_to_spirv(&mut self) -> Result<(), String> {
        let types = self
            .vars
            .iter()
            .filter(|(str, var)| var.ty.storage_class == StorageClass::PushConstant)
            .map(|(str, var)| (str.clone(), var.ty.clone()))
            .collect::<Vec<(String, MimirPtrType)>>();
        for (str, ty) in self
            .vars
            .iter()
            .filter(|(_, var)| var.ty.storage_class == StorageClass::PushConstant)
            .map(|(str, var)| (str.clone(), var.ty.clone()))
            .collect::<Vec<(String, MimirPtrType)>>()
        {
            if !self.ptr_types.contains_key(&ty.clone()) {
                let base_word = self.get_raw_type(&ty.base).unwrap();
                let ptr_ty = self
                    .spirv_builder
                    .type_pointer(None, ty.storage_class, base_word);

                self.ptr_types.insert(ty.clone(), ptr_ty);
            }

            let ptr_word = self.ptr_types.get(&ty).unwrap();

            let word = self
                .spirv_builder
                .variable(*ptr_word, None, ty.storage_class, None);

            self.spirv_builder.name(word, &str);

            self.vars.insert(
                str,
                MimirVariable {
                    ty: ty.clone(),
                    word: Some(word),
                },
            );
        }

        let indices: Vec<_> = types
            .iter()
            .map(|(param, _)| self.param_order.iter().position(|p| p == &param).unwrap())
            .collect();

        let mut ty_words = types
            .iter()
            .map(|_, ty| self.types.get(&ty.base).unwrap().clone())
            .collect::<Vec<Word>>();

        ty_words = indices.iter().map(|i| ty_words[*i]).collect();

        let push_const_struct_ty = self.spirv_builder.type_struct(ty_words);

        let ptr_push_const_struct_ty =
            self.spirv_builder
                .type_pointer(None, StorageClass::PushConstant, push_const_struct_ty);

        let push_const_var = self.spirv_builder.variable(
            ptr_push_const_struct_ty,
            None,
            StorageClass::PushConstant,
            None,
        );

        self.vars.insert(
            "push_const".to_string(),
            MimirVariable {
                ty: MimirPtrType {
                    base: MimirType::RuntimeArray(Box::new(MimirType::Float32)),
                    storage_class: StorageClass::PushConstant,
                },
                word: Some(push_const_var),
            },
        );
    }
}
