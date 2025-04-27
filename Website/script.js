document.addEventListener('DOMContentLoaded', function() {
    fetch('/backtest_data')
        .then(response => response.json())
        .then(data => {
            // Map data for Chart.js
            const timestamps = data.map(item => new Date(item.timestamp));
            const prices = data.map(item => item.price);
            const predictedPrices = data.map(item => item.predicted_price);
            const signals = data.map(item => item.signal);

            const ctx = document.getElementById('backtestChart').getContext('2d');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timestamps, // Use timestamps for the x-axis labels
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: prices,
                            borderColor: 'blue',
                            backgroundColor: 'rgba(0, 0, 255, 0.1)',
                            borderWidth: 1,
                            pointRadius: 0, // Hide points for a cleaner line
                            fill: false
                        },
                        {
                            label: 'Predicted Price',
                            data: predictedPrices,
                            borderColor: 'red',
                            backgroundColor: 'rgba(255, 0, 0, 0.1)',
                            borderWidth: 1,
                            pointRadius: 0, // Hide points
                            fill: false,
                            borderDash: [5, 5] // Dashed line for prediction
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false, // Allow chart to fill container width
                    scales: {
                        x: {
                            type: 'time', // Use time scale for timestamps
                            time: {
                                unit: 'minute', // Display unit as minutes
                                tooltipFormat: 'MMM d, yyyy HH:mm' // Custom tooltip format
                            },
                            title: {
                                display: true,
                                text: 'Timestamp'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Price (USD)'
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        // Format price as currency
                                        label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                    }
                                    // Add signal to tooltip for the actual price dataset
                                    if (context.dataset.label === 'Actual Price') {
                                        const index = context.dataIndex;
                                        label += ' | Signal: ' + signals[index];
                                    }
                                    return label;
                                }
                            }
                            /*
                            // Annotation plugin for BUY/SELL signals - currently not working correctly with time scale
                            // Keeping commented out for now, might require more complex logic or a different approach
                            annotation: {
                                annotations: signals.map((signal, index) => {
                                    if (signal === 'BUY') {
                                        return {
                                            type: 'point',
                                            xValue: timestamps[index], // Use timestamp for xValue
                                            yValue: prices[index],   // Use price for yValue
                                            backgroundColor: 'green',
                                            radius: 5,
                                            tooltip: { // Add tooltip for annotation
                                                enabled: true,
                                                content: 'BUY Signal',
                                                position: 'top'
                                            }
                                        };
                                    } else if (signal === 'SELL') {
                                         return {
                                            type: 'point',
                                            xValue: timestamps[index], // Use timestamp for xValue
                                            yValue: prices[index],   // Use price for yValue
                                            backgroundColor: 'red',
                                            radius: 5,
                                            tooltip: { // Add tooltip for annotation
                                                enabled: true,
                                                content: 'SELL Signal',
                                                position: 'top'
                                            }
                                        };
                                    }
                                    return null; // No annotation for HOLD
                                }).filter(annotation => annotation !== null) // Filter out null annotations
                            }
                            */
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching backtest data:', error));
});
