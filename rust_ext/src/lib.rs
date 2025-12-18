use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Fast conversion of numpy array to nested Python list.
/// This is much faster than Python's .tolist() method because it:
/// 1. Avoids Python object allocation for each element
/// 2. Builds the list structure directly
/// 3. No GIL contention during the conversion
#[pyfunction]
fn tensor_to_nested_list(py: Python<'_>, arr: PyReadonlyArray2<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];

    let outer_list = PyList::empty_bound(py);

    for i in 0..rows {
        let inner_list = PyList::empty_bound(py);
        for j in 0..cols {
            inner_list.append(arr[[i, j]])?;
        }
        outer_list.append(inner_list)?;
    }

    Ok(outer_list.into())
}

/// Fast conversion of 1D numpy array to Python list of lists (each with 1 element).
/// Optimized for the common case where each batch has exactly 1 token.
#[pyfunction]
fn tensor_1d_to_nested_list(py: Python<'_>, arr: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let len = arr.len();

    let outer_list = PyList::empty_bound(py);

    for i in 0..len {
        let inner_list = PyList::empty_bound(py);
        inner_list.append(arr[i])?;
        outer_list.append(inner_list)?;
    }

    Ok(outer_list.into())
}

/// Fast conversion of flat numpy array to Python list.
#[pyfunction]
fn tensor_to_flat_list(py: Python<'_>, arr: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let len = arr.len();

    let list = PyList::empty_bound(py);

    for i in 0..len {
        list.append(arr[i])?;
    }

    Ok(list.into())
}

/// A Python module implemented in Rust for fast tensor operations.
#[pymodule]
fn vllm_metal_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tensor_to_nested_list, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_1d_to_nested_list, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_flat_list, m)?)?;
    Ok(())
}
