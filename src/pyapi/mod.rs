//! PyO3 bindings exposed as `reducers._core`. Compiled only with the `python`
//! feature.

use pyo3::prelude::*;

mod reducers;
mod sigma_clip;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    reducers::register(m)?;
    sigma_clip::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
