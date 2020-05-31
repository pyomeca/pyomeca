Pyomeca's main functionality is to offer dedicated biomechanical routines.

<p align="center">
    <img src="/images/data-processing.svg" alt="api">
</p>

These features can be broadly grouped into different categories: filtering, normalization, matrix manipulation, signal processing and file output functions.

## Filters

Biomechanical data are inherently noisy.
And with noise, you will probably need filters.
Pyomeca implements the major types of Butterworth filters used in biomechanics.

!!! example
    === "Band pass"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.band_pass</div>

    === "Band stop"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.band_stop</div>

    === "High pass"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.high_pass</div>

    === "Low pass"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.low_pass</div>

## Normalization

It is common to use normalization procedures during biomechanical signal processing.
Pyomeca supports two types of normalization: signal normalization and time normalization.

!!! example
    === "Signal normalization"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.normalize</div>

    === "Time normalization"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.time_normalize</div>

## Matrix manipulation

The processing of biomechanical data often involves the use of matrix manipulation routines.
Some of them are implemented in Pyomeca.

!!! example
    === "Absolute value"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.abs</div>

    === "Center signal"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.center</div>

    === "Matmul"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.matmul</div>

    === "Norm"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.norm</div>

    === "RMS"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.rms</div>

    === "Square"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.square</div>

    === "Square root"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.sqrt</div>

## Signal processing

Pyomeca implements convenient and flexible functions to detect onsets and outliers, as well as to compute a Fourier Transform.

!!! example
    === "Onsets detection"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.detect_onset</div>

    === "Outliers detection"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.detect_outliers</div>

    === "FFT"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.fft</div>

## File output

While the [netcdf format](http://pyomeca.github.io/getting-started/#file-io) is the preferred file format for saving or sharing data structures, Pyomeca also supports writting csv and matlab files.
If you need more flexibility, the `to_wide_dataframe` method will allow you to use the pandas library to export your data in almost any existing formats.

!!! example
    === "Write csv file"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.to_csv</div>

    === "Write matlab file"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.to_matlab</div>

    === "Create wide pandas dataframe"
        <div class="template">/api/dataarray_accessor/#pyomeca.dataarray_accessor.DataArrayAccessor.to_wide_dataframe</div>
        
<script src="../js/template.js"></script>
<script>
    renderApiTemplate()
</script>
