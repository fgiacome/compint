use pyo3::prelude::*;

pub mod game;
pub mod simulations;

/// A Python module implemented in Rust.
#[pymodule]
fn quixo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulations::parallel_simulation_py, m)?)?;
    Ok(())
}
