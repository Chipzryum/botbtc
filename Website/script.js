document.addEventListener('DOMContentLoaded', function() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const metricsGrid = document.getElementById('metricsGrid');
    const loadingMetrics = document.getElementById('loadingMetrics');

    loadingMessage.style.display = 'block';
    errorMessage.style.display = 'none';

    // Load main backtest results
    let backtestDataPromise = fetch('/backtest_data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        });
    
    // Load chart data separately
    let chartDataPromise = fetch('/price_chart_data')
        .then(response => {
            if (!response.ok) {
                // If chart data is not available, we'll handle this gracefully
                if (response.status === 404) {
                    console.warn("Price chart data not found. Charts may be limited.");
                    return { chart_data: [] };
                }
                throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        })
        .catch(error => {
            console.warn("Error loading price chart data:", error);
            return { chart_data: [] }; // Return empty data on error
        });

    // Process both data sources
    Promise.all([backtestDataPromise, chartDataPromise])
        .then(([backtestData, chartData]) => {
            loadingMessage.style.display = 'none';
            loadingMetrics.style.display = 'none';

            // --- 1. Display Metrics ---
            displayMetrics(backtestData.metrics);
            
            // --- 2. Display Trade Summary if available ---
            if (backtestData.trades_summary) {
                displayTradeSummary(backtestData.trades_summary);
            }

            // --- 3. Prepare Chart Data ---
            const priceData = chartData.chart_data || [];
            const trades = backtestData.trades || [];
            const equityCurveData = backtestData.equity_curve || [];

            // --- 4. Create Trade Annotations ---
            const tradeAnnotations = createTradeAnnotations(trades);

            // --- 5. Render Price Chart if we have data ---
            if (priceData.length > 0) {
                const timestamps = priceData.map(item => new Date(item.timestamp));
                const prices = priceData.map(item => item.price);
                renderPriceChart(timestamps, prices, tradeAnnotations);
            } else {
                console.warn("No price data available for charting.");
                const priceCanvas = document.getElementById('backtestChart');
                if (priceCanvas) {
                    const ctx = priceCanvas.getContext('2d');
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Price chart data not available.", priceCanvas.width / 2, priceCanvas.height / 2);
                }
            }

            // --- 6. Render Equity Curve Chart ---
            if (equityCurveData.length > 0) {
                renderEquityCurveChart(equityCurveData);
            } else {
                console.warn("No equity curve data available.");
                const equityCanvas = document.getElementById('equityChart');
                if (equityCanvas) {
                    const ctx = equityCanvas.getContext('2d');
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Equity curve data not available.", equityCanvas.width / 2, equityCanvas.height / 2);
                }
            }
        })
        .catch(error => {
            console.error('Error fetching or processing backtest data:', error);
            loadingMessage.style.display = 'none';
            loadingMetrics.style.display = 'none';
            errorMessage.textContent = `Error loading backtest data: ${error.message}. Please ensure the backtest was run and the server is working.`;
            errorMessage.style.display = 'block';
        });

    // --- Helper Functions ---
    
    // New function to display trade summary information
    function displayTradeSummary(tradeSummary) {
        // Check if we have a container for the trade summary
        let summaryContainer = document.getElementById('tradeSummary');
        if (!summaryContainer) {
            // Create it if it doesn't exist
            summaryContainer = document.createElement('div');
            summaryContainer.id = 'tradeSummary';
            summaryContainer.classList.add('trade-summary');
            
            // Insert after metrics but before charts
            const chartContainer = document.querySelector('.chart-container');
            if (chartContainer && chartContainer.parentNode) {
                chartContainer.parentNode.insertBefore(summaryContainer, chartContainer);
            } else {
                document.body.appendChild(summaryContainer);
            }
        }
        
        summaryContainer.innerHTML = `
            <h3>Additional Trade Summary</h3>
            <p>The backtest includes ${tradeSummary.count} additional trades not shown in detail.</p>
            <div class="summary-stats">
                <div>Total PnL: ${tradeSummary.total_pnl.toFixed(2)}</div>
                <div>Winning Trades: ${tradeSummary.winning_trades}</div>
                <div>Losing Trades: ${tradeSummary.losing_trades}</div>
                <div>Time Range: ${tradeSummary.time_range}</div>
            </div>
        `;
    }

    function displayMetrics(metrics) {
        metricsGrid.innerHTML = ''; // Clear loading/previous metrics
        if (!metrics || Object.keys(metrics).length === 0) {
            metricsGrid.innerHTML = '<p>No metrics available.</p>';
            return;
        }
        for (const [key, value] of Object.entries(metrics)) {
            const metricItem = document.createElement('div');
            metricItem.classList.add('metric-item');

            const label = document.createElement('span');
            label.classList.add('metric-label');
            label.textContent = key;

            const val = document.createElement('span');
            val.classList.add('metric-value');
            // Add specific formatting if needed (e.g., for currency)
            if (key.toLowerCase().includes('capital') || key.toLowerCase().includes('profit') || key.toLowerCase().includes('pnl')) {
                 val.textContent = typeof value === 'number' ? value.toLocaleString('en-US', { style: 'currency', currency: 'USD' }) : value;
            } else if (key.toLowerCase().includes('%')) {
                 val.textContent = typeof value === 'number' ? `${value.toFixed(2)}%` : value;
            }
             else {
                val.textContent = value;
            }


            metricItem.appendChild(label);
            metricItem.appendChild(val);
            metricsGrid.appendChild(metricItem);
        }
    }

    function createTradeAnnotations(trades) {
        const annotations = {};
        trades.forEach((trade, index) => {
            // Entry annotation
            annotations[`entry_${index}`] = {
                type: 'point',
                xValue: new Date(trade['Entry Time']).valueOf(), // Use timestamp value
                yValue: trade['Entry Price'],
                backgroundColor: trade.Direction === 'Long' ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)', // Green for Long, Red for Short
                borderColor: trade.Direction === 'Long' ? 'darkgreen' : 'darkred',
                borderWidth: 1,
                radius: 6,
                label: { // Use label instead of tooltip for newer annotation plugin versions
                    content: `${trade.Direction} Entry @ ${trade['Entry Price'].toFixed(2)}`,
                    display: true,
                    position: 'start', // Adjust as needed ('start', 'end', 'center')
                    yAdjust: trade.Direction === 'Long' ? -15 : 15, // Position label above/below point
                    backgroundColor: 'rgba(0,0,0,0.6)',
                    color: 'white',
                    font: { size: 10 },
                    padding: 3,
                    borderRadius: 3,
                }
            };
            // Exit annotation
            annotations[`exit_${index}`] = {
                type: 'point',
                xValue: new Date(trade['Exit Time']).valueOf(), // Use timestamp value
                yValue: trade['Exit Price'],
                backgroundColor: 'rgba(100, 100, 100, 0.7)', // Grey for exit
                borderColor: 'black',
                borderWidth: 1,
                radius: 5,
                borderDash: [6, 6], // Dashed circle for exit
                 label: {
                    content: `Exit @ ${trade['Exit Price'].toFixed(2)} (PnL: ${trade.PnL.toFixed(2)})`,
                    display: true,
                    position: 'start',
                    yAdjust: trade.Direction === 'Long' ? 15 : -15,
                    backgroundColor: 'rgba(0,0,0,0.6)',
                    color: 'white',
                    font: { size: 10 },
                    padding: 3,
                    borderRadius: 3,
                }
            };
             // Line connecting entry and exit
            annotations[`line_${index}`] = {
                type: 'line',
                xMin: new Date(trade['Entry Time']).valueOf(),
                xMax: new Date(trade['Exit Time']).valueOf(),
                yMin: trade['Entry Price'],
                yMax: trade['Exit Price'],
                borderColor: trade.PnL >= 0 ? 'rgba(0, 128, 0, 0.5)' : 'rgba(255, 0, 0, 0.5)', // Green for profit, Red for loss
                borderWidth: 1.5,
                borderDash: [3, 3]
            };
        });
        return annotations;
    }


    function renderPriceChart(timestamps, prices, tradeAnnotations) {
        const ctx = document.getElementById('backtestChart').getContext('2d');
        if (!ctx) {
            console.error("Price chart canvas not found");
            return;
        }

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: 'Close Price',
                    data: prices,
                    borderColor: 'rgba(0, 123, 255, 0.8)', // Bootstrap primary blue
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0, // No points on the main line
                    fill: false,
                    tension: 0.1 // Slight smoothing
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute', // Adjust based on your data frequency
                            tooltipFormat: 'MMM d, yyyy HH:mm:ss',
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'MMM d HH:mm',
                                day: 'MMM d'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Timestamp'
                        },
                        ticks: {
                             autoSkip: true,
                             maxTicksLimit: 20 // Limit ticks for readability
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (USD)'
                        },
                        ticks: {
                             callback: function(value, index, values) {
                                return '$' + value.toLocaleString(); // Format as currency
                            }
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
                                    label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    },
                    annotation: { // Annotation plugin configuration
                        annotations: tradeAnnotations
                    },
                    legend: {
                        display: true // Show legend for 'Close Price'
                    }
                },
                 interaction: { // Improve hover/tooltip interaction
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

     function renderEquityCurveChart(equityData) {
        const ctx = document.getElementById('equityChart').getContext('2d');
         if (!ctx) {
            console.error("Equity chart canvas not found");
            return;
        }

        const equityTimestamps = equityData.map(item => new Date(item.timestamp));
        const equityValues = equityData.map(item => item.equity);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: equityTimestamps,
                datasets: [{
                    label: 'Equity Curve',
                    data: equityValues,
                    borderColor: 'rgba(40, 167, 69, 0.8)', // Bootstrap success green
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true, // Fill area under the curve
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                 scales: {
                    x: {
                        type: 'time',
                         time: {
                            unit: 'day', // Adjust based on backtest duration
                            tooltipFormat: 'MMM d, yyyy HH:mm',
                             displayFormats: {
                                minute: 'HH:mm',
                                hour: 'MMM d HH:mm',
                                day: 'MMM d, yyyy'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Timestamp'
                        },
                         ticks: {
                             autoSkip: true,
                             maxTicksLimit: 15
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Portfolio Equity (USD)'
                        },
                         ticks: {
                             callback: function(value, index, values) {
                                return '$' + value.toLocaleString(); // Format as currency
                            }
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
                                    label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    },
                     legend: {
                        display: false // Hide legend if only one dataset
                    }
                },
                 interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

}); // End DOMContentLoaded