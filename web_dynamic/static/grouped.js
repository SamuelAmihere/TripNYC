function load_Boroughs_loc_id() {
    const url = '/taxi_zone_lookup';
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: $${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (!Array.isArray(data)) {
                console.error('Data is not an array:', data);
                throw new Error('Data is not an array');
            }
            
            const boroughs = data
                .filter((ob) => Object.keys(ob))


                datafinal = []
                data.forEach(function(d) {
                    Object.values(d).forEach(function(v) {
                        datafinal.push(v[1]);
                    });
                });
                // console.log('Final==', datafinal);

            return datafinal;
        })
        .catch(error => {
            console.error('Error in load_Boroughs_loc_id:', error);
            throw error; // re-throw to allow caller to handle
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

function getBoroughPaths(svg_doc, boroughs) {

    if (Array.isArray(boroughs) && boroughs.length == 1) {
        console.log('Working on a single boroughs');
        // Select borough group if only one borough is passed
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
        
        return [paths_arr, total_price];
    }

    var borough_paths_arr = [];
    for (let i = 0; i < boroughs.length; i++) {
        const borough = boroughs[i];
        // Select borough group
        const borough_group = svg_doc.select("#borough_" + borough);
        // select all paths in Manhattan group by id: e.g Manhattan_area_3 
        const borough_paths = borough_group.selectAll("path[id*='" + borough + "_area_']");
        var paths_arr = {};
        for (let i = 0; i < borough_paths.nodes().length; i++) {
            // generate random price for each borough
            borough_paths_arr.push(borough_paths.nodes()[i].id)
        }
    }
    return borough_paths_arr;
}


function displayLocID(Id) {
    if (Id == undefined) {
        return;
    }
    console.log('ID:', Id);
}


document.getElementById('svg-object').addEventListener('load', function() {
    // changge background color of svg
    

    // console.log(locations);
    
    const svgDoc = this.contentDocument;
    
    const svg = d3.select(svgDoc).select("svg");

    // select all groups with borough in id name

    // ----Select Manhattan group-----
    const manhattan = svg.select("#borough_Manhattan");

    var locations = {}
    load_Boroughs_loc_id().then(function(result) {
        
        const pathsofboroughs = getBoroughPaths(svg, ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']);
        // place each value in results in each path in pathsofboroughs
        for (let i = 0; i < pathsofboroughs.length; i++) {

            const path = svg.node().querySelector("#" + pathsofboroughs[i]);
            if (path == null || path == undefined) {
                continue;
            }
            path.style.fill = "green";
            path.style.fillOpacity = result[i] / 100;
            // add event listener to each path: click
            path.addEventListener("click", function() {
                displayLocID(result[i])
            });
        }
        
        console.log('<<<Loaded>>>', pathsofboroughs);
        // console.log('Paths of Boroughs', pathsofboroughs);
    });




    // Change color of Manhattan paths based on price
    const borough_p = getBoroughPaths(svg, "Manhattan");
    const kv = borough_p[0];
    const total_price = borough_p[1];

    for (const [key, value] of Object.entries(kv)) {
        
        const path = svg.node().querySelector("#" + key);
        path.style.fill = "green";
        path.style.fillOpacity = value;
        // add event listener to each path: display price on hover
        path.addEventListener("mouseover", function() {
            displayLocID(value * total_price)
        });
        path.addEventListener("mouseout", function() {
            removeID();
        });
    }
    manhattan.node().children[76].style.fill = "red";
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