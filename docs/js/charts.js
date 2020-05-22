function gridData(nRows, nCols, width) {
  let out = [];
  d3.range(nRows).forEach((row) => {
    d3.range(nCols).forEach((col) => {
      out.push({ x: width * (col % nCols), y: width * (row % nRows), width });
    });
  });
  return out;
}

function drawMatrix(id, matrixDimensions, matrixLabels, titleLabel) {
  const rectangleWidth = 40,
    skewAngle = 45,
    spacing = 10,
    to_radian = (degree) => (degree * Math.PI) / 180,
    scaling = 0.5,
    xUnit = rectangleWidth * matrixDimensions[0],
    yUnit = rectangleWidth * matrixDimensions[1],
    zUnit = rectangleWidth * matrixDimensions[2];

  let dimensions = {
    width: document.getElementById(id).clientWidth,
    margin: { left: 80, bottom: 10, right: 80, top: 10 },
  };
  dimensions.height =
    dimensions.margin.top +
    dimensions.margin.bottom +
    xUnit +
    zUnit * scaling +
    10 * spacing;
  dimensions.innerHeight =
    dimensions.height - dimensions.margin.top - dimensions.margin.bottom;
  dimensions.innerWidth =
    dimensions.width - dimensions.margin.left - dimensions.margin.right;

  const wrapper = d3
    .select(`#${id}`)
    .append("svg")
    .attr("width", dimensions.width)
    .attr("height", dimensions.height);

  const bounds = wrapper
    .append("g")
    .style(
      "transform",
      `translate(${(dimensions.innerWidth + zUnit * scaling - yUnit) / 2}px, ${
        (dimensions.innerHeight + zUnit * scaling - xUnit + 4 * spacing) / 2
      }px)`
    );

  const matrix = bounds.append("g").attr("class", "matrix-stroke");

  matrix
    .append("g")
    .selectAll(".grid")
    .data(gridData(matrixDimensions[0], matrixDimensions[1], rectangleWidth))
    .enter()
    .append("rect")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y)
    .attr("width", (d) => d.width)
    .attr("height", (d) => d.width)
    .style("fill", "#ECEFF1");

  matrix
    .append("g")
    .attr(
      "transform",
      `skewX(-${skewAngle}) scale(1, ${scaling}) translate(0, -${zUnit})`
    )
    .selectAll(".grid")
    .data(gridData(matrixDimensions[2], matrixDimensions[1], rectangleWidth))
    .enter()
    .append("rect")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y)
    .attr("width", (d) => d.width)
    .attr("height", (d) => d.width)
    .style("fill", "#CFD8DC");

  matrix
    .append("g")
    .attr(
      "transform",
      `translate(${yUnit}, 0) skewY(-${skewAngle}) scale(${scaling}, 1)`
    )
    .selectAll(".row")
    .data(gridData(matrixDimensions[0], matrixDimensions[2], rectangleWidth))
    .enter()
    .append("rect")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y)
    .attr("width", (d) => d.width)
    .attr("height", (d) => d.width)
    .style("fill", "#B0BEC5");

  const lines = matrix.append("g");

  lines
    .append("line")
    .attr("x1", -spacing)
    .attr("x2", -spacing)
    .attr("y1", 0)
    .attr("y2", xUnit);

  lines
    .append("line")
    .attr("x1", 0)
    .attr("x2", yUnit)
    .attr("y1", xUnit + spacing)
    .attr("y2", xUnit + spacing);

  const ySpacing = spacing * Math.sin(to_radian(skewAngle));
  const xSpacing = Math.sqrt(spacing ** 2 - ySpacing ** 2);
  lines
    .append("line")
    .attr("x1", yUnit + xSpacing)
    .attr("x2", yUnit + xSpacing + zUnit * scaling)
    .attr("y1", xUnit + ySpacing)
    .attr("y2", xUnit + ySpacing - zUnit * scaling);

  const texts = bounds.append("g").attr("class", "chart-text");

  texts
    .append("text")
    .style("text-anchor", "end")
    .attr("dx", -spacing)
    .attr("x", -spacing)
    .attr("y", xUnit / 2)
    .text(matrixLabels[0]);

  texts
    .append("text")
    .style("text-anchor", "middle")
    .style("alignment-baseline", "hanging")
    .attr("dy", spacing)
    .attr("x", yUnit / 2)
    .attr("y", xUnit + spacing)
    .text(matrixLabels[1]);

  texts
    .append("text")
    .style("text-anchor", "middle")
    .style("alignment-baseline", "hanging")
    .attr("dy", 10)
    .attr(
      "transform",
      `translate(${(2 * (yUnit + xSpacing) + zUnit * scaling) / 2}, ${
        (2 * (xUnit + ySpacing) - zUnit * scaling) / 2
      }) rotate(-${skewAngle})`
    )
    .text(matrixLabels[2]);

  const title = texts
    .append("g")
    .attr("transform", `translate(-200, ${-zUnit * scaling - 2 * spacing})`);

  title
    .append("text")
    .style("alignment-baseline", "ideographic")
    .style("font-weight", "bold")
    .attr("y", -spacing)
    .text(`${titleLabel} matrix`);

  title
    .append("text")
    .style("alignment-baseline", "hanging")
    .attr("y", -0.5 * spacing)
    .text(
      `Example with ${matrixDimensions[0]} ${matrixLabels[0]}, ${matrixDimensions[1]} ${matrixLabels[1]} and ${matrixDimensions[2]} ${matrixLabels[2]}:`
    );
}

async function drawApi(id) {
  const apiData = await d3.json("../api/api.json");

  const hierarchy = d3
    .hierarchy(apiData)
    .sum((d) => d.value)
    .sort((a, b) => a.height - b.height || a.value - b.value);

  const spacing = 10;
  const fontSize = 12;
  let dimensions = {
    width: document.getElementById(id).clientWidth,
    margin: { left: 0, bottom: spacing, right: 0, top: spacing },
  };
  dimensions.height = (fontSize + spacing * 2) * hierarchy.value;
  dimensions.innerHeight =
    dimensions.height - dimensions.margin.top - dimensions.margin.bottom;
  dimensions.innerWidth =
    dimensions.width - dimensions.margin.left - dimensions.margin.right;

  const partition = d3
    .partition()
    .size([dimensions.innerHeight, dimensions.innerWidth])
    .padding(1.5)(hierarchy);

  const colorScale = d3.scaleOrdinal([
    "#F44336",
    "#E91E63",
    "#2196F3",
    "#4CAF50",
    "#673AB7",
  ]);

  const wrapper = d3
    .select(`#${id}`)
    .append("svg")
    .attr("width", dimensions.width)
    .attr("height", dimensions.height);

  const bounds = wrapper
    .append("g")
    .style(
      "transform",
      `translate(${dimensions.margin.left}px, ${dimensions.margin.top}px)`
    );

  const cell = bounds
    .selectAll("a")
    .data(partition.descendants())
    .join("a")
    .attr("transform", (d) => `translate(${d.y0},${d.x0})`)
    .attr("href", (d) => d.data.link)
    .attr("target", "_blank")
    .attr("class", "cell");

  cell
    .append("rect")
    .attr("width", (d) => d.y1 - d.y0)
    .attr("height", (d) => d.x1 - d.x0)
    .attr("fill-opacity", 0.3)
    .attr("fill", (d) => {
      if (!d.depth) return "#ccc";
      while (d.depth > 1) d = d.parent;
      return colorScale(d.data.name);
    });

  const texts = cell
    .append("g")
    .attr("class", "chart-text")
    .attr("transform", `translate(${spacing}, ${spacing})`);

  texts
    .append("text")
    .attr("dominant-baseline", "hanging")
    .text((d) => d.data.name);

  // ------ Tooltip -------
  const tooltip = d3.select(`#tooltip`);

  const onMouseLeave = () => {
    tooltip.style("display", "none");
  };

  const onMouseEnter = (datum) => {
    tooltip.select("#tooltip-title").text(datum.data.name);
    tooltip.select("#tooltip-docstring").html(datum.data.docstring);
    tooltip.style("display", "block");
  };

  cell.on("mouseenter", onMouseEnter).on("mouseleave", onMouseLeave);
}
