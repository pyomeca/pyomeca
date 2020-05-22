The starting point for working with Pyomeca is to create an object with one of the specific methods associated with the different classes available:

<p align="center">
    <img src="/images/object-creation.svg" alt="api">
</p>

Pyomeca offers several ways to create these objects: from scratch, from random data, from files or from other data structures.

## From scratch

The first way to create a data array in Pyomeca is to directly specify the data.

!!! example
    === "Angles"
        <div class="template">/api/angles/#pyomeca.angles.Angles</div>
    
    === "Markers"
        <div class="template">/api/markers/#pyomeca.markers.Markers</div>

    === "Rototrans" 
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans</div>

    === "Analogs"
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs</div>

## From random data

We occasionally want to quickly create an object to test implementations or prototype new features.
In this case, we could simply use random numerical values.
Pyomeca offers a method for directly creating objects from random data.

!!! Example
    === "Angles"
        <div class="template">/api/angles/#pyomeca.angles.Angles.from_random_data</div>
        
    === "Markers"
        <div class="template">/api/markers/#pyomeca.markers.Markers.from_random_data</div>

    === "Rototrans" 
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_random_data</div>

    === "Analogs"
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_random_data</div>

## From files

Most of the time, we want to create objects from files collected during experimentation.
Pyomeca supports most of the formats used in biomechanics.

!!! note
    Pyomeca does not support a format you need?
    You can inform us by opening an [issue](https://github.com/romainmartinez/pyomeca/issues) or even submit a [pull request](https://github.com/romainmartinez/pyomeca/pulls) to make your implementation available to the whole community!

=== "c3d"
    !!! Example
        === "Markers"
            <div class="template">/api/markers/#pyomeca.markers.Markers.from_c3d</div>
            
        === "Analogs"
            <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_c3d</div>

=== "csv"
    !!! Example
        === "Markers"
            <div class="template">/api/markers/#pyomeca.markers.Markers.from_csv</div>
            
        === "Analogs"
            <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_csv</div>

=== "excel"
    !!! Example
        === "Markers"
            <div class="template">/api/markers/#pyomeca.markers.Markers.from_excel</div>
            
        === "Analogs"
            <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_excel</div>

=== "mot"
    !!! Example
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_mot</div>

=== "trc"
    !!! Example
        <div class="template">/api/markers/#pyomeca.markers.Markers.from_trc</div>

=== "sto"
    !!! Example
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_sto</div>

## From other data structures

We often have to switch between different representations of the same data.
Pyomeca implements different matrix manipulation routines such as getting Euler angles or a marker to/from a rototranslation matrix.

### Angles & Rototrans

!!! Example
    === "Angles from Rototrans"
        <div class="template">/api/angles/#pyomeca.angles.Angles.from_rototrans</div>

    === "Rototrans from Angles"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_euler_angles</div>

### Markers & Rototrans

!!! Example
    ===! "Markers from Rototrans"
        <div class="template">/api/markers/#pyomeca.markers.Markers.from_rototrans</div>

    === "Rototrans from Markers"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_markers</div>

### Processed Rototrans

!!! Example
    ===! "Rototrans from a transposed Rototrans"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_transposed_rototrans</div>

    === "Rototrans from an averaged Rototrans"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_averaged_rototrans</div>
        
<script src="../js/template.js"></script>
<script>
    renderApiTemplate()
</script>
