
from html import escape
from pathlib import Path
import collections
import statistics
import time

import json
from pathlib import Path
from .interface import VisualizerInterface
from rag_eval_flow.utils.data_utils import find_latest_metric_files, construct_one_model_metrics_df, sample_dataframe, assemble_accurate_df

class HTMLVisualizer(VisualizerInterface):
    @property
    def name(self) -> str:
        return "Interactive Model-Level HTML Report"

    @property
    def description(self) -> str:
        return "Generates an HTML report with sliders for filtering metric scores for a single model."
    
    def _prepare_html_data(self, latest_files : list[Path]) -> tuple[list, list, list]:
        print(f"Using latest files: {[file.stem for file in latest_files]}")

        merged_df, discovered_metrics = construct_one_model_metrics_df(latest_files)
        sorted_df = assemble_accurate_df(merged_df) # removes false positives and negatives for content score stability
        print("DF columns: ", sorted_df.columns)
        rows = sorted_df.to_dict(orient='records')
        categories = sorted(list(discovered_metrics))

        dimensions = []
        for metric in categories:
            dimensions.append({
                'name': metric.replace('_', ' ').title(),
                'key': f'{metric}_score',
                'filter_id': f'{metric}_score'
            })
        
        return dimensions, categories, rows


    def generate(self, input_path: Path, output_dir: Path) -> Path:
        print(f"--- Running '{self.name}' for model '{input_path.name}' ---")

        latest_files = sorted(find_latest_metric_files(str(input_path)))
        if not latest_files:
            print("  No metric files found, skipping report generation.")
            return output_dir / f"{input_path.name}_report_empty.html"

        dimensions, categories, rows = self._prepare_html_data(latest_files)
        
        # Define the output path and generate the report
        report_path = output_dir / f"{input_path.name}_report.html"

        distributions = {}
        mean_scores = {}
        
        for dim in dimensions:
            distribution = collections.Counter()
            scores = []
            
            for row in rows:
                try:
                    score_value = float(row.get(dim['key'], 0) or 0)
                    score = int(score_value)
                    distribution[score] += 1
                    scores.append(score_value)
                except (ValueError, TypeError):
                    pass
            
            # Calculate mean score if we have scores
            if scores:
                mean_scores[dim['key']] = round(statistics.mean(scores), 2)
            else:
                mean_scores[dim['key']] = 0
            
            # Fill in missing scores with 0
            for score in range(-1, 6):  # -1 to 5
                if score not in distribution:
                    distribution[score] = 0
                    
            distributions[dim['key']] = {
                'labels': list(range(-1, 6)),  # -1 to 5
                'counts': [distribution[i] for i in range(-1, 6)]
            }
        
        # Start building the HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text Analysis Results</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/15.6.1/nouislider.min.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1 {
                    text-align: center;
                    color: #333;
                    margin-bottom: 10px;
                }
                .text-card {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    overflow: hidden;
                    transition: opacity 0.3s ease;
                }
                .text-card.hidden {
                    display: none;
                }
                .text-container {
                    display: flex;
                    flex-direction: row;
                    flex-wrap: wrap;
                }
                .text-section {
                    flex: 1;
                    min-width: 300px;
                    padding: 20px;
                    border-right: 1px solid #eee;
                }
                .scores-section {
                    flex: 2;
                    padding: 20px;
                }
                .score-category {
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 1px solid #eee;
                }
                .score-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .score-title {
                    font-weight: bold;
                    color: #444;
                }
                .score-value {
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 15px;
                    color: white;
                    background-color: #4CAF50;
                }
                .score-reasoning {
                    margin-bottom: 10px;
                    color: #555;
                }
                .poor { background-color: #f44336; }
                .fair { background-color: #FF9800; }
                .good { background-color: #4CAF50; }
                .excellent { background-color: #2196F3; }
                .summary-section {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
                .ground-truth, .model-output, .difficulty, .response-type {
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    border-left: 3px solid #2196F3;
                }
                .score-category h3 {
                    margin-top: 0;
                    margin-bottom: 15px;
                    color: #333;
                }
                .overall-quality {
                    background-color: rgba(33, 150, 243, 0.1);
                    border-left: 5px solid #2196F3;
                    padding: 15px;
                    margin-bottom: 25px;
                    border-radius: 5px;
                }
                h1 .sort-info {
                    font-size: 0.7em;
                    font-weight: normal;
                    color: #666;
                    display: block;
                    margin-top: 5px;
                }
                /* Filter Controls Styling */
                .filter-container {
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .filter-title {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                }
                .filter-controls {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .filter-row {
                    display: flex;
                    flex-wrap: wrap;
                    align-items: center;
                    gap: 15px;
                    margin-bottom: 25px;
                    padding-bottom: 15px;
                    border-bottom: 1px solid #eee;
                }
                .filter-label {
                    font-weight: bold;
                    font-size: 16px;
                    min-width: 200px;
                    display: block;
                    margin-bottom: 10px;
                }
                .slider-container {
                    flex: 1;
                    min-width: 300px;
                    height: 50px;
                    position: relative;
                }
                .range-slider {
                    width: 100%;
                    height: 10px;
                    margin: 20px 0;
                    position: relative;
                }
                .slider-values {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 10px;
                    font-size: 14px;
                    color: #666;
                }
                .min-filter-value, .max-filter-value {
                    font-weight: bold;
                    background-color: #2196F3;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 15px;
                    margin-left: 10px;
                    min-width: 30px;
                    text-align: center;
                    display: inline-block;
                }
                .filter-count {
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 5px;
                    font-size: 14px;
                    margin-top: 15px;
                }
                /* Range Slider Styles */
                .noUi-target {
                    background: #d3d3d3;
                    border-radius: 5px;
                    border: none;
                    box-shadow: none;
                }
                .noUi-connects {
                    background: #d3d3d3;
                    border-radius: 5px;
                }
                .noUi-connect {
                    background: #2196F3;
                }
                .noUi-handle {
                    height: 24px !important;
                    width: 24px !important;
                    border-radius: 50%;
                    background: #2196F3;
                    box-shadow: 0 0 5px rgba(0,0,0,0.2);
                    border: 2px solid white;
                    cursor: pointer;
                }
                .noUi-handle:before, .noUi-handle:after {
                    display: none;
                }
                /* Chart container styles */
                .charts-container {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    padding: 20px;
                }
                .chart-title {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                    text-align: center;
                }
                .chart-wrapper {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
                    gap: 20px;
                }
                .chart-item {
                    flex: 1;
                    min-width: 300px;
                    max-width: 500px;
                    margin-bottom: 20px;
                }
                .mean-score-label {
                    color: #d32f2f;
                    font-weight: bold;
                    text-align: center;
                    margin-top: 10px;
                    font-size: 14px;
                }
                /* Checkbox Styles for Category Filters */
                .checkbox-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-top: 10px;
                }
                .checkbox-item {
                    display: flex;
                    align-items: center;
                    margin-right: 15px;
                }
                .checkbox-item input[type="checkbox"] {
                    margin-right: 5px;
                }
                .checkbox-item label {
                    cursor: pointer;
                    font-size: 14px;
                }
                .keyword-filter-section {
                    padding-bottom: 15px;
                    margin-bottom: 15px;
                    border-bottom: 1px solid #eee;
                }
                /* Checkbox active state */
                .checkbox-item input[type="checkbox"]:checked + label {
                    font-weight: bold;
                    color: #2196F3;
                }
                /* Clear filters button */
                .clear-filters {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-left: 15px;
                    font-size: 14px;
                }
                .clear-filters:hover {
                    background-color: #d32f2f;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Text Analysis Results <span class="sort-info">Sorted by Entry ID</span></h1>
        """
        
        # Add score distribution charts section with properly formatted mean scores
        html_content += """
                <!-- Score Distribution Charts -->
                <div class="charts-container">
                    <div class="chart-title">Score Distribution Analysis</div>
                    <div class="chart-wrapper">
        """
        
        # Add chart items for each dimension
        for dim in dimensions:
            html_content += f"""
                        <div class="chart-item">
                            <canvas id="{dim['filter_id']}Chart"></canvas>
                            <div class="mean-score-label">Mean Score: {mean_scores[dim['key']]}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
        """
        
        # Add filter controls
        html_content += """
                <!-- Filter Controls -->
                <div class="filter-container">
                    <div class="filter-title">Filter Entries</div>
                    <div class="filter-controls">
        """
        
        # Add Category keyword filter checkboxes
        html_content += """
                        <!-- Category Filters -->
                        <div class="keyword-filter-section">
                            <div class="filter-label">Filter by Category:</div>
                            <div class="checkbox-container">
        """
        
        # Add checkboxes for each category
        for category in categories:
            display_name = category.replace('_', ' ').title()
            html_content += f"""
                                <div class="checkbox-item">
                                    <input type="checkbox" id="{category}-checkbox" name="category-filter" value="{category}">
                                    <label for="{category}-checkbox">{display_name}</label>
                                </div>
            """
        
        html_content += """
                            </div>
                        </div>
        """
        
        # Add a filter slider for each dimension
        for dim in dimensions:
            html_content += f"""
                        <div class="filter-row" id="{dim['filter_id']}-filter-row">
                            <div class="filter-label">{dim['name']} Score: <span id="min-{dim['filter_id']}-value" class="min-filter-value">-1</span> - <span id="max-{dim['filter_id']}-value" class="max-filter-value">10</span></div>
                        <div class="slider-container">
                                <div id="{dim['filter_id']}-slider" class="range-slider"></div>
                            <div class="slider-values">
                                <span>-1</span>
                                <span>0</span>
                                <span>1</span>
                                <span>2</span>
                                <span>3</span>
                                <span>4</span>
                                <span>5</span>
                            </div>
                        </div>
                        </div>
            """
        
        html_content += """
                    </div>
                    <button id="clear-filters" class="clear-filters">Clear All Filters</button>
                    <div class="filter-count">Showing <span id="visible-count">0</span> of <span id="total-count">0</span> entries</div>
                </div>
        """
        
        # Function to determine score class
        def get_score_class(score):
            try:
                score_value = float(score)
                if score_value == -1:
                    return "poor"  # Special handling for -1
                elif score_value < 3:
                    return "poor"
                elif score_value < 4:
                    return "fair"
                elif score_value < 5:
                    return "good"
                else:
                    return "excellent"
            except:
                return "good"  # Default
        
        # Add this helper function at the top of your file
        def sanitize_attr_name(name):
            """Convert attribute name to HTML-valid format by replacing spaces with hyphens"""
            return name.replace(' ', '-')

        # Add each text entry's results to the HTML
        for row in rows:
            # Create data attributes for all dimensions to enable filtering
            data_attrs = ' '.join([f'data-{sanitize_attr_name(dim["filter_id"])}="{row.get(dim["key"], "0")}"' for dim in dimensions])
            
            html_content += f"""
            <div class="text-card" {data_attrs}>
                <div class="text-container">
                    <div class="text-section">
                        <h2>Entry ID: {escape(str(row.get('orig_index', 'N/A')))}</h2>
                        
                        <div class="Query">
                            <h3>Query</h3>
                            <p>{escape(str(row.get('question', 'Not provided')))}</p>
                        </div>
                        
                        <div class="model-output">
                            <h3>SLM Response</h3>
                            <p>{escape(str(row.get('model_answer', 'Not provided')))}</p>
                        </div>

                        <div class="ground-truth">
                            <h3>Ground Truth Answer</h3>
                            <p>{escape(str(row.get('answer', 'Not provided')))}</p>
                        </div>

                        <div class="difficulty">
                            <h3>Difficulty</h3>
                            <p>{escape(str(row.get('difficulty', 'Not provided')))}</p>
                        </div>

                        <div class="response-type">
                            <h3>Response Type</h3>
                            <p>{escape(str(row.get('response_type', 'Not provided')))}</p>
                        </div>

                    </div>
                    
                    <div class="scores-section">
                        <h2>Evaluation Scores & Reasoning</h2>
            """
            
            # Add all score categories
            for dim in dimensions:
                score = row.get(dim['key'], "N/A")
                score_class = get_score_class(score)
                reasoning_key = f"{dim['key'][:-6]}_explanation"
                
                # First dimension gets highlighted styling
                category_class = "score-category overall-quality" if dim == dimensions[0] else "score-category"
                
                html_content += f"""
                        <div class="{category_class}">
                            <h3>{dim['name']}</h3>
                            <div class="score-header">
                                <span class="score-title">Score:</span>
                                <span class="score-value {score_class}">{score}</span>
                            </div>
                            <div class="score-reasoning">
                                <strong>Judge Explanation:</strong> {escape(str(row.get(reasoning_key, 'Not provided')))}
                            </div>
                        </div>
                """
            
            # Add summary assessment section if available
            if 'summary_assessment' in row and row['summary_assessment']:
                html_content += f"""
                        <div class="summary-section">
                            <h3>Summary Assessment</h3>
                            <p>{escape(row['summary_assessment'])}</p>
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
            </div>
            """
        
        # Add JavaScript for chart creation
        chart_data = {}
        for dim in dimensions:
            chart_data[dim['filter_id']] = distributions[dim['key']]
        
        # Convert chart_data to JSON
        chart_data_json = json.dumps(chart_data)
        
        # Create JavaScript content for the charts and filtering
        js_content = f"""
            <script src="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/15.6.1/nouislider.min.js"></script>
            <script>
                // Chart data
                const chartData = {chart_data_json};
                
                // Mean scores
                const meanScores = {{
                    {', '.join([f'"{dim["filter_id"]}": {mean_scores[dim["key"]]}' for dim in dimensions])}
                }};
                
                // Categories for filtering
                const categories = {json.dumps(categories)};
                
                // Store chart references
                const chartRefs = {{}};
                
                // Create charts
                document.addEventListener('DOMContentLoaded', function() {{
                    // Initialize charts - removing the problematic line
                    const dimensionsConfig = [
                        {', '.join([f'{{ id: "{dim["filter_id"]}", name: "{dim["name"]}" }}' for dim in dimensions])}
                    ];
                    
                    dimensionsConfig.forEach(dim => {{
                        const chartElement = document.getElementById(`${{dim.id}}Chart`);
                        if (!chartElement) {{
                            console.error(`Chart element not found: ${{dim.id}}Chart`);
                            return;
                        }}
                        
                        const chartCtx = chartElement.getContext('2d');
                        try {{
                            chartRefs[dim.id] = new Chart(chartCtx, {{
                                type: 'bar',
                                data: {{
                                    labels: chartData[dim.id].labels,
                                    datasets: [{{
                                        label: dim.name,
                                        data: chartData[dim.id].counts,
                                        backgroundColor: [
                                            '#8b0000', // Score -1 - Not applicable/invalid
                                            '#f44336', // Score 0 - Poor
                                            '#f5554a', // Score 1
                                            '#f98b86', // Score 2
                                            '#fcd2d0', // Score 3
                                            '#a5d6a7', // Score 4
                                            '#4caf50'  // Score 5 - Excellent
                                        ],
                                        borderWidth: 1
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    plugins: {{
                                        legend: {{
                                            display: false
                                        }},
                                        title: {{
                                            display: true,
                                            text: `${{dim.name}} Score Distribution`
                                        }},
                                        annotation: {{
                                            annotations: {{
                                                meanLine: {{
                                                    type: 'line',
                                                    xMin: meanScores[dim.id],
                                                    xMax: meanScores[dim.id],
                                                    borderColor: '#d32f2f',
                                                    borderWidth: 2,
                                                    borderDash: [5, 5],
                                                    label: {{
                                                        display: true,
                                                        content: 'Mean: ' + meanScores[dim.id],
                                                        position: 'top'
                                                    }}
                                                }}
                                            }}
                                        }}
                                    }},
                                    scales: {{
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Number of Entries'
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'Score'
                                            }}
                                        }}
                                    }}
                                }}
                            }});
                            console.log(`Chart created for ${{dim.id}}`);
                        }} catch (err) {{
                            console.error(`Error creating chart for ${{dim.id}}: `, err);
                        }}
                    }});
                    
                    // Function to calculate score distributions for visible entries
                    function calculateDistributions() {{
                        const visibleCards = document.querySelectorAll('.text-card:not(.hidden)');
                        
                        // Initialize distribution counters for each dimension (0-10)
                        const distributions = {{}};
                        const scores = {{}};
                        
                        dimensionsConfig.forEach(dim => {{
                            distributions[dim.id] = Array(7).fill(0);  // Counts for scores -1 to 5
                            scores[dim.id] = [];
                        }});
                        
                        // Count scores for each visible card
                        visibleCards.forEach(card => {{
                            dimensionsConfig.forEach(dim => {{
                                // Convert dim.id to a valid HTML attribute name
                                const attrName = dim.id.replace(/ /g, '-');
                                const score = parseInt(card.getAttribute(`data-${{attrName}}`) || 0);
                                if (score >= -1 && score <= 5) {{
                                    distributions[dim.id][score + 1]++;  // +1 adjustment for -1 index
                                    scores[dim.id].push(score);
                                }}
                            }});
                        }});
                        
                        // Calculate means
                        const meanScores = {{}};
                        dimensionsConfig.forEach(dim => {{
                            meanScores[dim.id] = scores[dim.id].length > 0 ? 
                                scores[dim.id].reduce((a, b) => a + b, 0) / scores[dim.id].length : 0;
                            meanScores[dim.id] = Math.round(meanScores[dim.id] * 100) / 100;
                        }});
                        
                        return {{ distributions, meanScores }};
                    }}
                    
                    // Function to update charts with new distribution data
                    function updateCharts(distributions, meanScores) {{
                        dimensionsConfig.forEach(dim => {{
                            const chart = chartRefs[dim.id];
                            if (chart) {{
                                chart.data.datasets[0].data = distributions[dim.id];
                                chart.options.plugins.annotation.annotations.meanLine.xMin = meanScores[dim.id];
                                chart.options.plugins.annotation.annotations.meanLine.xMax = meanScores[dim.id];
                                chart.options.plugins.annotation.annotations.meanLine.label.content = 'Mean: ' + meanScores[dim.id];
                                chart.update();
                                
                                // Update mean score label in HTML
                                const chartContainer = document.getElementById(`${{dim.id}}Chart`).closest('.chart-item');
                                const meanLabel = chartContainer.querySelector('.mean-score-label');
                                if (meanLabel) {{
                                    meanLabel.textContent = 'Mean Score: ' + meanScores[dim.id];
                                }}
                            }}
                        }});
                    }}
                    
                    // Initialize filter sliders with error handling
                    const filterConfig = dimensionsConfig.map(dim => {{
                        return {{ 
                            id: dim.id, 
                            dataAttr: `data-${{dim.id}}` 
                        }};
                    }});
                    
                    // Create all filter sliders
                    filterConfig.forEach(config => {{
                        try {{
                            console.log('Creating slider for: ' + config.id);
                            const sliderElement = document.getElementById(`${{config.id}}-slider`);
                            if (!sliderElement) {{
                                console.error(`Element not found: ${{config.id}}-slider`);
                                return;
                            }}
                            
                            noUiSlider.create(sliderElement, {{
                                start: [-1, 5],
                                connect: true,
                                step: 1,
                                range: {{
                                    'min': -1,
                                    'max': 5
                                }}
                            }});
                            
                            // Add update event
                            sliderElement.noUiSlider.on('update', function() {{
                                updateFilters();
                            }});
                            
                            console.log(`Slider created successfully for ${{config.id}}`);
                        }} catch (error) {{
                            console.error(`Error creating slider for ${{config.id}}: ` + error.message);
                        }}
                    }});
                    
                    // Check all category checkboxes by default
                    categories.forEach(category => {{
                        const checkbox = document.getElementById(`${{category}}-checkbox`);
                        if (checkbox) {{
                            checkbox.checked = true;
                            checkbox.addEventListener('change', function() {{
                                updateFilters();
                            }});
                        }}
                    }});
                    
                    // Clear filters button
                    const clearFiltersButton = document.getElementById('clear-filters');
                    if (clearFiltersButton) {{
                        clearFiltersButton.addEventListener('click', function() {{
                            // Reset all checkboxes
                            categories.forEach(category => {{
                                const checkbox = document.getElementById(`${{category}}-checkbox`);
                                if (checkbox) {{
                                    checkbox.checked = true;
                                }}
                            }});
                            
                            // Reset all sliders
                            filterConfig.forEach(config => {{
                                const slider = document.getElementById(`${{config.id}}-slider`);
                                if (slider && slider.noUiSlider) {{
                                    slider.noUiSlider.set([0, 5]);
                                }}
                            }});
                            
                            // Update the filters to show all entries
                            updateFilters();
                        }});
                    }}
                    
                    // Function to update all filters
                    function updateFilters() {{
                        try {{
                            const textCards = document.querySelectorAll('.text-card');
                            const visibleCount = document.getElementById('visible-count');
                            const totalCount = document.getElementById('total-count');
                            
                            // Get current filter values for scores
                            const scoreFilters = filterConfig.map(config => {{
                                try {{
                                    const slider = document.getElementById(`${{config.id}}-slider`);
                                    if (!slider || !slider.noUiSlider) {{
                                        console.warn(`Slider not initialized for ${{config.id}}`);
                                        return {{
                                            dataAttr: config.dataAttr,
                                            min: 0,
                                            max: 5
                                        }};
                                    }}
                                    
                                    const values = slider.noUiSlider.get();
                                    const min = Math.round(parseFloat(values[0]));
                                    const max = Math.round(parseFloat(values[1]));
                                    
                                    // Update filter display values
                                    const minElement = document.getElementById(`min-${{config.id}}-value`);
                                    const maxElement = document.getElementById(`max-${{config.id}}-value`);
                                    
                                    if (minElement) minElement.textContent = min;
                                    if (maxElement) maxElement.textContent = max;
                                    
                                    return {{
                                        dataAttr: config.dataAttr.replace(/ /g, '-'),
                                        min: min,
                                        max: max
                                    }};
                                }} catch (error) {{
                                    console.error(`Error getting filter values for ${{config.id}}: ` + error.message);
                                    return {{
                                        dataAttr: config.dataAttr,
                                        min: 0,
                                        max: 5
                                    }};
                                }}
                            }});
                            
                            // Get checked categories
                            const selectedCategories = [];
                            categories.forEach(category => {{
                                const checkbox = document.getElementById(`${{category}}-checkbox`);
                                if (checkbox && checkbox.checked) {{
                                    selectedCategories.push(category);
                                }}
                            }});
                            
                            // Initialize counters
                            if (totalCount) totalCount.textContent = textCards.length;
                            let visibleCards = 0;
                            
                            // Filter text cards based on score filters
                            textCards.forEach(card => {{
                                let passesScoreFilters = true;
                                
                                // Check score filters
                                scoreFilters.forEach(filter => {{
                                    const quality = parseFloat(card.getAttribute(filter.dataAttr) || 0);
                                    if (quality < filter.min || quality > filter.max) {{
                                        passesScoreFilters = false;
                                    }}
                                }});
                                
                                // Apply visibility
                                if (passesScoreFilters) {{
                                    card.classList.remove('hidden');
                                    visibleCards++;
                                }} else {{
                                    card.classList.add('hidden');
                                }}
                            }});
                            
                            if (visibleCount) visibleCount.textContent = visibleCards;
                            
                            // Update charts based on visible entries
                            const {{ distributions, meanScores }} = calculateDistributions();
                            updateCharts(distributions, meanScores);
                        }} catch (error) {{
                            console.error('Error in updateFilters: ' + error.message);
                        }}
                    }}
                    
                    // Initialize view
                    setTimeout(function() {{
                        updateFilters();
                        console.log('Filters initialized');
                    }}, 100);
                }});
            </script>
        """
        
        # Close the HTML
        html_content += js_content
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write the HTML to file
        report_path = output_dir / f"{time.strftime('%m%d')}{input_path.name}_report.html"

        with open(report_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        print(f"HTML report generated successfully at {report_path}")
        return report_path
