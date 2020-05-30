## Working with labelled multidimensional arrays

Pyomeca introduces a concise interface to read, analyse, visualize and plot biomechanical data.

Such data are typically *multi-dimensional*, such as joint angles with associated axes, degrees of freedom and time frames.

<img class="center" src="/images/objects/angles.svg"></img>

[NumPy](https:numpy.org) is the fundamental package for multi-dimensional computing with Python.
While NumPy provides an efficient data structure and an intuitive interface, biomechanical datasets typically contain more than just raw numbers and have labels which encode how the array values map to different dimensions such as axes, degrees of freedom, channels or time frames.

Pyomeca is built upon and extends the core strengths of [xarray](http://xarray.pydata.org/en/stable/index.html), which keeps tracks of labels and provides a powerful and concise interface which makes it easy to:

-   Apply any operations over dimensions by name (`array.sum(dim="time")`) instead of an arbitrary axis (`array.sum(axis=2)`).
-   Select values by labels (`array.sel(axis="x")` or `emg.sel(channel="biceps")`).
-   Vectorize computation across multiple dimensions.
-   Use the [split-apply-combine](https://vita.had.co.nz/papers/plyr.pdf) paradigm, for example: `emg.groupby("channel").mean()` or any custom function: `emg.groupby('channel').map(lambda x: x - x.mean())`).
-   Keep track of metadata in the `array.attrs` Python dictionary (`array.attrs["rate"]`).
-   Extent the xarray interface with domain specific functionalities with custom accessors on xarray objects. In pyomeca, the biomechanics specific functions are registered under the `meca` name space (`array.meca`).

Working with labels makes it much easier to work with multi-dimensional arrays as you do not have to keep track of the order of the dimensions or insert dummy dimensions to align arrays.
This allows for a more intuitive, more concise, and less error-prone developer experience.

!!! note
    As the underlying data structure is still a NumPy array, NumPy functions (`np.abs(array)`) and indexing (`array[:, 0, 1]`) work out of the box.

By leveraging xarray data structures, Pyomeca inherits their features such as built-in [interpolation](http://xarray.pydata.org/en/stable/interpolation.html), [computation](http://xarray.pydata.org/en/stable/computation.html), [GroupBy](http://xarray.pydata.org/en/stable/groupby.html), [data wrangling](http://xarray.pydata.org/en/stable/combining.html), [parallel computing](http://xarray.pydata.org/en/stable/dask.html) and [plotting](http://xarray.pydata.org/en/stable/plotting.html).

!!! info "Extending xarray"
    Xarray is designed as a general-purpose library and tries to avoid including domain specific functionalities.
    But inevitably, the need for more domain specific logic arises.
    That's why Pyomeca and [dozens of other scientific packages](http://xarray.pydata.org/en/stable/related-projects.html) extend xarray.

    Extending data structure in Python is usually achieved with class inheritance.
    However inheritance is not very robust for large class such as `xarray.DataArray`.
    To add domain specific functionality, pyomeca follows xarray developers' recommendations and use a custom "accessor".
    For more information, you can check out the [xarray documentation](http://xarray.pydata.org/en/stable/internals.html#extending-xarray).

## Core functionalities

Pyomeca has four data structures built upon [xarray](http://xarray.pydata.org/en/stable/index.html).
Each structure is associated with a specific biomechanical data type and has specialized functionalities:

| Class | Dimensions | Description |
|-------------------------|-------------------------------------|------------------------------------------------------------------------|
| [`Analogs`](/api/analogs/#pyomeca.analogs.Analogs) | `("channel", "time")` | Generic signals such as EMGs, force signals or any other analog signals |
| [`Angles`](/api/angles/#pyomeca.angles.Angles) | `("axis", "channel", "time")` | Joint angles |
| [`Markers`](/api/markers/#pyomeca.markers.Markers) | `("axis", "channel", "time")` | Skin marker positions |
| [`Rototrans`](/api/rototrans/#pyomeca.rototrans.Rototrans) | `("row", "col", "time")` | Rototranslation matrix |

While there are technically dozens of functions in pyomeca one can generally group them into two distinct categories:

1.  [Object creation](https://pyomeca.github.io/object-creation/) with the `from_*` methods. For example, if you want to define a marker array from a csv file: `markers = Markers.from_csv(...)`.
2.  [Data processing](https://pyomeca.github.io/data-processing/) with the `meca` array accessor. For example, to low-pass filter our previous markers: `markers.meca.low_pass(...)`.

!!! note
    Check out the API reference to see the parameters, use cases and examples associated with each function.
    
You can explore all of pyomeca's public API on the following interactive visualization. 
Hover the mouse over any block to display a short description with some examples
 and click to jump to the corresponding API reference. 

<div id="api-exploration">
    <div id="tooltip" class="admonition info tooltip">
        <p id="tooltip-title" class="admonition-title"></p>
        <p id="tooltip-docstring"></p>
    </div>
</div>

<script src="https://d3js.org/d3.v5.min.js"></script>
<script src="../js/charts.js"></script>
<script>
    drawApi("api-exploration");
</script>
