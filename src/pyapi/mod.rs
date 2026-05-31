//! PyO3 bindings exposed as `reducers._core`. Compiled only with the `python`
//! feature.

use pyo3::prelude::*;

mod reducers;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    reducers::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
