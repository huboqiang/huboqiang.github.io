---
layout: page
title: humanretina
header: humanretina
group: navigation
---
{% include JB/setup %}

Type the name of genes you are interested in:
--------------------------------------------

<label for="inputGene">Input Genes:</label>
<input type="text" id="autocomplete">
<button onclick="PlotGene()">Submit</button>
<!-- Plotly chart will be drawn inside this DIV -->


<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="http://cdnjs.cloudflare.com/ajax/libs/jquery.tipsy/1.0.2/jquery.tipsy.css" rel="stylesheet" type="text/css" />
        <script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.1/jquery.js"></script>
        <script type="text/javascript" src="http://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js"></script>
        <!--<script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/d3/3.4.11/d3.min.js"></script>-->
        <script type="text/javascript" src="http://d3js.org/d3.v3.js"></script>
        <script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/jquery.tipsy/1.0.2/jquery.tipsy.min.js"></script>
        <script type="text/javascript" src="http://qcloud-1252801552.file.myqcloud.com/plotly-latest.min.js"></script>
        <link href="http://cdn.datatables.net/1.10.12/css/jquery.dataTables.min.css" rel="stylesheet"></link>
        <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
        <script src="http://libs.baidu.com/bootstrap/3.0.3/js/bootstrap.min.js"></script>
        <script type="text/javascript" src="http://d3js.org/d3.v3.min.js"></script>
        <script type="text/javascript" src="http://qcloud-1252801552.file.myqcloud.com/geneName.js"></script>
    <script>
        $('#autocomplete').autocomplete({
        source: function (req, responseFn) {
        var term = $.ui.autocomplete.escapeRegex(req.term),
        matcher = new RegExp('^' + term, 'i'),
        matches = $.grep(options, function (item) {
            return matcher.test(item);
        });
        responseFn(matches.slice(0, 10));
         }
    });
</script>

<div id="boxPlotStage" class="boxPlotStage"></div>
<div id="main" style="background-color:#FFFFFF;height=40%;width:60%;float:left;">
        <div id="myDiv0"></div>
    </div>

<div id="sup" style="background-color:#FFFFFF;height=40%;width:40%;float:left;">
        <div id="myDiv1"></div>
        <div id="myDiv2"></div>
</div>

<script>
        function PlotGene(){
        var URL_BASE = "http://198.13.42.241:5000/gene/";
        var URL_BASE_BOX = "http://198.13.42.241:5000/geneBox/";
        function update_url() {
              return URL_BASE + document.getElementById("autocomplete").value;
        }
        function updateBox_url() {
              return URL_BASE_BOX + document.getElementById("autocomplete").value;
        }


        length = 100
        colorList = d3.scale.linear().domain([1, length])
            .interpolate(d3.interpolateHcl)
            .range([d3.rgb("#B1B1B1"), d3.rgb('#2200FF')]);

        Plotly.d3.csv(update_url(), function(err, rows) {
            var Types = ["Amacrine", "Bipolar", "Blood", "Fibroblast", "Horizontal", "Microglia", "Muller", "Photoreceptor", "RGC", "RPC", "RPE", 'Undef'];
            var Stages = ['7W', '8W', '9W', '9WP', '10W', '11W', '12WP', '15W', '19W', '25W', '26W', '27W'];
            M_colorListType = {
                "Amacrine": "#8dd3c7",
                "Bipolar": "#ffffb3",
                "Blood": "#bebada",
                "Fibroblast": "#fb8072",
                "Horizontal": "#80b1d3",
                "Microglia": "#fdb462",
                "Muller": "#b3de69",
                "Photoreceptor": "#fccde5",
                "RGC": "#d9d9d9",
                "RPC": "#bc80bd",
                "RPE": "#ccebc5",
                "Undef": "#333333",

            }
            M_colorListStage = {
              "7W" :   "#5E4FA2",
              "8W" :   "#3682BA",
              "9W" :   "#5CB7A9",
              "9WP" :   "#98D5A4",
              "10W" :   "#D0EC9C",
              "11W" :   "#F3FAAD",
              "12WP" :   "#FEF0A7",
              "15W" :   "#FDCD7B",
              "19W" :   "#FA9C58",
              "25W" :   "#EE6445",
              "26W" :   "#D0384D",
              "27W" :   "#9E0142"
            }

            function unpack(rows, key) {
                return rows.map(function(row) {
                    return row[key];
                });
            }

            function TPMColor(strinput) {
                var TPMMax = 1000;
                var v0 = parseFloat(strinput, 2);
                var v1 = Math.log2(v0 / 100 + 1);
                var v2 = Math.log2(TPMMax / 100 + 1);
                idx = parseInt((v1 / v2) * 100);
                if (idx >= 100) {
                    idx = 99;
                }
                return colorList(idx);
            }
            var dataTPM = Types.map(function(type) {
                var rowsFiltered = rows.filter(function(row) {
                    return (row.type === type);
                });
                return {
                    mode: 'markers',
                    x: unpack(rowsFiltered, 'PC1'),
                    y: unpack(rowsFiltered, 'PC2'),
                    text: unpack(rowsFiltered, 'sam'),
                    xaxis: 'x1',
                    yaxis: 'y1',
                    marker: {
                        sizemode: 'area',
                        size: 5,
                        color: unpack(rowsFiltered, 'TPM').map(TPMColor),
                    }
                };
            });

            var dataType = Types.map(function(type) {
                var rowsFiltered = rows.filter(function(row) {
                    return (row.type === type);
                });
                return {
                    mode: 'markers',
                    name: type,
                    x: unpack(rowsFiltered, 'PC1'),
                    y: unpack(rowsFiltered, 'PC2'),
                    text: unpack(rowsFiltered, 'sam'),
                    xaxis: 'x1',
                    yaxis: 'y1',
                    marker: {
                        sizemode: 'area',
                        size: 5,
                    }
                };
            });
            var dataStage = Stages.map(function(stage) {
                var rowsFiltered = rows.filter(function(row) {
                    return (row.stage === stage);
                });
                return {
                    mode: 'markers',
                    name: stage,
                    x: unpack(rowsFiltered, 'PC1'),
                    y: unpack(rowsFiltered, 'PC2'),
                    text: unpack(rowsFiltered, 'sam'),
                    xaxis: 'x2',
                    yaxis: 'y2',
                    marker: {
                        symbol: 'circle',
                        size: 5,
                        color: M_colorListStage[unpack(rowsFiltered, 'stage')[0]],
                    }
                };
            });
            var layout0 = {
                xaxis1: {
                    domain: [0, 1],
                    anchor: "x2",
                    title: 'PC1'
                },
                yaxis1: {
                    domain: [0, 1],
                    anchor: "y2",
                    title: 'PC2',
                },
                margin: {
                    l: 30,
                    r: 30,
                    b: 30,
                    t: 30,
                    pad: 4
                },
                hovermode: 'closest',
                height: 800
            };
            var layout1 = {
                xaxis1: {
                    domain: [0, 1],
                    anchor: "x0",
                    title: 'PC1'
                },
                yaxis1: {
                    domain: [0, 1],
                    anchor: "y1",
                    title: 'PC2',
                },
                margin: {
                    l: 30,
                    r: 30,
                    b: 30,
                    t: 30,
                    pad: 4
                },
                hovermode: 'closest',
                height: 400
            };
            var layout2 = {
                xaxis1: {
                    domain: [0, 1],
                    anchor: "x2",
                    title: 'PC1'
                },
                yaxis1: {
                    domain: [0, 1],
                    anchor: "y2",
                    title: 'PC2',
                },
                margin: {
                    l: 30,
                    r: 30,
                    b: 30,
                    t: 30,
                    pad: 4
                },
                hovermode: 'closest',
                height: 400
            };
            Plotly.newPlot('myDiv0', dataTPM, layout0, {
                showLink: false
            });
            Plotly.newPlot('myDiv1', dataType, layout1, {
                showLink: false
            });
            Plotly.newPlot('myDiv2', dataStage, layout2, {
                showLink: false
            });
        });





        /* Boxplot */
          var urlBox = updateBox_url();
          d3.select("#boxPlotStage").selectAll("*").remove();
          var json = (function() {
              var json = null;
              $.ajax({
                  'async': false,
                  'global': false,
                  'url': urlBox,
                  'dataType': "json",
                  'success': function(data) {
                      json = data;
                  }
              });
              return json;
          })();
          var data = json['results'];
          var layout = {
              yaxis: {
                  title: 'TPM',
                  zeroline: false
              },
              boxmode: 'group'
          };
          Plotly.newPlot('boxPlotStage', data, layout);
                }

</script>

</div> <!-- doc-container -->

</div><!-- div id="wrap" -->

<div id="footer">
      <div class="container bs-docs-bar-footer">
          <p><a href="https://github.com/huboqiang">Fork me on Github</a> &copy; 2017 by Boqiang Hu. </p>
      </div>
    </div>

    



<!-- Latest compiled and minified JavaScript, requires jQuery 1.x (2.x not supported in IE8) -->
<!-- Placed at the end of the document so the pages load faster -->
<script src="/assets/themes/bootstrap-3/bootstrap-3/jquery-1.11.2.min.js"></script>
    <script src="/assets/themes/bootstrap-3/bootstrap-3/bootstrap-3.3.1/js/bootstrap.min.js"></script>
    <script src="/assets/themes/bootstrap-3/bootstrap-3/js/application.js"></script>
    <script src="/assets/themes/bootstrap-3/bootstrap-3/google-code-prettify/prettify.js"></script>
    <script type="text/javascript">
      $(document).ready(function() {
        /*google-code-prettify*/
        $('pre').addClass('prettyprint').attr('style', 'overflow:auto');
        prettyPrint();
        $('pre .language-tips').parent().addClass('tom-callout-tips');
        /* A link icon on the left of headers. */
        $(".bs-docs-container h1,.bs-docs-container h2, .bs-docs-container h3, .bs-docs-container h4, .bs-docs-container h5, .bs-docs-container h6").each(function(i, el) {
          var $el, icon, id;
          $el = $(el);
          id = $el.attr('id');
          icon = '<span class="glyphicon glyphicon-link"></span>';
          if (id) {
             $el.prepend($("<a class='anchor' />").attr("href", "#" + id).html(icon));
          }
        })
      });
</script>
    
<script> 


