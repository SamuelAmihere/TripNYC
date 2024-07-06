function load_Boroughs_loc_id(borough) {
    const lup = 'Data/Taxi Zone Maps and Lookup Tables/taxi_zone_lookup.csv';
    
    return d3.csv(lup).then(function(data) {
        var boroughs = [];
        data.forEach(function(d) {
            if (d.Borough == borough) {
                boroughs.push(d.LocationID);
            }
        });
        return boroughs;
    });
}

function getBoroughs(svg_doc) {
    // select all groups with borough in id name
    const boroughs = svg_doc.selectAll("g[id*='borough_']");

    var boroughs_arr = {}
    for (let i = 0; i < boroughs.nodes().length; i++) {
        // generate random price for each borough
        boroughs_arr[boroughs.nodes()[i].id.split("_")[1]] = Math.floor(Math.random() * 100) + 1;
        // boroughs_arr.push(boroughs.nodes()[i].id.split("_")[1]);
    }
    
    return boroughs_arr;
}

function getBoroughPaths(svg_doc, borough) {

    // Select Manhattan group
    const borough_group = svg_doc.select("#borough_" + borough);
    // select all paths in Manhattan group by id: e.g Manhattan_area_3 
    const borough_paths = borough_group.selectAll("path[id*='" + borough + "_area_']");
    var paths_arr = {};
    for (let i = 0; i < borough_paths.nodes().length; i++) {
        // generate random price for each borough
        paths_arr[borough_paths.nodes()[i].id] = Math.floor(Math.random() * 100) + 1;
    }
    // weight the paths by price
    const total_price = Object.values(paths_arr).reduce((a, b) => a + b, 0);
    for (const [key, value] of Object.entries(paths_arr)) {
        paths_arr[key] = value / total_price;
    }
    
    load_Boroughs_loc_id(borough).then(function(result) {
        // Select ids starting with text_ 
        const pattern = [result.map(id => "#text_" + id)];
        const locations_in_borough = svg_doc.selectAll(pattern[0]);
        
        console.log(locations_in_borough);

    });
    return [paths_arr, total_price];



}


function displayPrice(price) {
    const price_div = document.createElement("div");
    price_div.id = "price-div";
    price_div.style.position = "absolute";
    price_div.style.top = event.clientY + "px";
    price_div.style.left = event.clientX + "px";
    price_div.style.backgroundColor = "black";
    price_div.style.color = "white";
    price_div.style.padding = "5px";
    price_div.style.borderRadius = "5px";
    price_div.style.zIndex = "1000";
    price_div.innerHTML = "Price: $" + price.toFixed(2);
    document.body.appendChild(price_div);
    // setTimeout(() => {
    //     price_div.remove();
    // }, 2000);
}
function removePrice() {
    const price_div = document.getElementById("price-div");
    if (price_div){
        price_div.remove();
    }
}



document.getElementById('svg-object').addEventListener('load', function() {
    var locations = {}
    
    load_Boroughs_loc_id('Manhattan').then(function(result) {
        // console.log(result);
        locations['Manhattan'] = result;
    });
    // console.log(locations);
    
    const svgDoc = this.contentDocument;
    
    const svg = d3.select(svgDoc).select("svg");

    // select all groups with borough in id name

    // ----Select Manhattan group-----
    const manhattan = svg.select("#borough_Manhattan");
    console.log(manhattan.node().children);

    // Change color of Manhattan paths based on price
    const borough_p = getBoroughPaths(svg, "Manhattan");
    const kv = borough_p[0];
    const total_price = borough_p[1];

    for (const [key, value] of Object.entries(kv)) {
        
        manhattan.node().querySelector("#" + key).style.fill = "green";
        // manhattan.node().querySelector("#" + key).style.fillOpacity = value*10;
        // add event listener to each path: display price on hover
        manhattan.node().querySelector("#" + key).addEventListener("mouseover", function() {
            displayPrice(value*total_price);
        });
        manhattan.node().querySelector("#" + key).addEventListener("mouseout", function() {
            removePrice();
        });

    }
    // manhattan.node().children[76].style.fill = "red";
    // manhattan.node().children[76].style.fillOpacity = 0.8;

    // console.log(manhattan.node().children[0].style.fill);

    // Pulse animation for Manhattan
    function pulseManhattan() {
        manhattan
            .transition()
            .duration(1000)
            .style("fill-opacity", 0.7)
            .transition()
            .duration(1000)
            .style("fill-opacity", 0.3)
            .on("end", pulseManhattan);  // Repeat the animation
    }

    pulseManhattan();

    // Fade in other boroughs
    svg.selectAll("path")
        .filter(function() {
            return !this.parentNode.id.includes("borough_Manhattan");
        })
        .style("opacity", 0)
        .transition()
        .delay(1000)
        .duration(2000)
        .style("opacity", 1);
});