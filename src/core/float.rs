use num_traits::float::FloatCore;
use num_traits::{FromPrimitive, Signed};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::AddAssign;

// A Float trait that captures the requirements we need for the various places
// we need floats. These requirements are imposed by ndarray and kiddo
pub trait AdriannFloat:
    FloatCore
    + Debug
    + Default
    + AddAssign
    + Serialize
    + for<'de> Deserialize<'de>
    + Signed
    + Copy
    + Sync
    + Send
    + FromPrimitive
{
}

impl AdriannFloat for f32 {}
impl AdriannFloat for f64 {}
