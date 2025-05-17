use std::any::Any;


pub trait MimirArray<T: Copy + Sized + 'static> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<T>() * self.len()
    }

    fn from_iter<I>(iter:I) -> Self where I: IntoIterator<Item = T>, Self: Sized;

    #[doc(hidden)]
    fn get_inner_as_any(&self) -> &dyn Any;

    fn to_iter(&self) -> Box<dyn Iterator<Item = T>>;
}