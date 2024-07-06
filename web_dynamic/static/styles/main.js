
document.getElementById('svg-object').addEventListener('load', function() {

    const svgDoc = this.contentDocument;
    const svg = d3.select(svgDoc).select("svg");

    // select all groups with borough in id name
    console.log(svgDoc);

    // ----Select Manhattan group-----
    const manhattan = svg.select("#borough_Manhattan");
    // console.log(manhattan);

    // ----Select all paths in Manhattan group-----
    const paths = manhattan.selectAll("path");
    // console.log(paths);

    // ----Select all text in Manhattan group-----
    const text = manhattan.selectAll("text");
    // console.log(text);

    // ----Select all text in Manhattan group-----
    const circles = manhattan.selectAll("circle");
    // console.log(circles);

    // ----Select all text in Manhattan group-----
    const rects = manhattan.selectAll("rect");
    // console.log(rects);

    // ----Select all text in Manhattan group-----
    const polygons = manhattan.selectAll("polygon");
    // console.log(polygons);
});
