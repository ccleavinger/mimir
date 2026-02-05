use std::hash::{DefaultHasher, Hash, Hasher};

use mimir_ir::{compiler_err, ir::MimirTyVar, util::error::ASTError};
use small_uid::SmallUid;

use crate::compiler::kernel_compiler;

pub(crate) fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

impl kernel_compiler::Compiler {
    pub(crate) fn add_ty_var(&mut self, name: &String, ty_var: MimirTyVar) -> Result<(), ASTError> {
        let uuid = self.create_uuid(name)?;

        self.map.insert(uuid, ty_var);

        Ok(())
    }

    pub(crate) fn create_uuid(&mut self, name: &String) -> Result<u64, ASTError> {
        let scoped_name = if self.scope_stack.len() > 1 {
            let curr_scope = self.scope_stack.last().unwrap();

            // rust compiler can't support `"` in variable names \
            // so using it to specify scope prevents colliding w/ other user-defined variables
            &format!("{name}\"{curr_scope}\"")
        } else {
            // if we don't have any scoping occurring (just in the kernel no ifs, loops, or other stuff)
            name
        };

        let hash = calculate_hash(scoped_name);

        if self.hash_to_uuid.contains_key(&hash) {
            return compiler_err!("{name} is a duplicated variable");
        }

        // smaller size but higher likelihood of collisions (shouldn't be problematic for Mimir)
        let uid = SmallUid::new();

        self.hash_to_uuid.insert(hash, uid.0);
        self.uuid_to_name.insert(uid.0, scoped_name.clone());

        Ok(uid.0)
    }

    pub(crate) fn get_ty_var(&self, name: &String) -> Option<&MimirTyVar> {
        if let Some(uuid) = self.get_uuid(name) {
            return self.map.get(uuid);
        }
        None
    }

    pub(crate) fn get_uuid(&self, name: &String) -> Option<&u64> {
        {
            let hash = calculate_hash(name);

            if let Some(uuid) = self.hash_to_uuid.get(&hash) {
                return Some(uuid);
            }
        }

        if self.scope_stack.len() == 1 {
            return None;
        }

        for scope in self.scope_stack[1..].iter() {
            let scoped_name = format!("{name}\"{scope}\"");
            let hash = calculate_hash(&scoped_name);

            if let Some(uuid) = self.hash_to_uuid.get(&hash) {
                return Some(uuid);
            }
        }

        None
    }
}
