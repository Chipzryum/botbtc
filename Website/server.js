const express = require('express');
const path = require('path');
const fs = require('fs');
const zlib = require('zlib');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the Website directory
app.use(express.static(path.join(__dirname)));

// Endpoint to serve backtest results
app.get('/backtest_data', (req, res) => {
    // Check for compressed file first
    const compressedPath = path.join(__dirname, 'backtest_results.json.gz');
    const regularPath = path.join(__dirname, 'backtest_results.json');
    
    // Try compressed file first
    if (fs.existsSync(compressedPath)) {
        fs.readFile(compressedPath, (err, data) => {
            if (err) {
                console.error('Error reading compressed backtest data:', err);
                return res.status(500).send('Error reading backtest results');
            }
            
            // Decompress and send
            zlib.gunzip(data, (err, decompressed) => {
                if (err) {
                    console.error('Error decompressing backtest data:', err);
                    return res.status(500).send('Error processing backtest results');
                }
                
                res.setHeader('Content-Type', 'application/json');
                res.send(decompressed);
            });
        });
    }
    // Fall back to regular JSON file
    else if (fs.existsSync(regularPath)) {
        fs.readFile(regularPath, 'utf8', (err, data) => {
            if (err) {
                console.error('Error reading backtest data:', err);
                return res.status(500).send('Error reading backtest results');
            }
            
            res.setHeader('Content-Type', 'application/json');
            res.send(data);
        });
    } else {
        res.status(404).send('Backtest results not found');
    }
});

// Endpoint to serve price chart data separately
app.get('/price_chart_data', (req, res) => {
    const chartDataPath = path.join(__dirname, 'price_chart_data.json');
    
    if (fs.existsSync(chartDataPath)) {
        fs.readFile(chartDataPath, 'utf8', (err, data) => {
            if (err) {
                console.error('Error reading chart data:', err);
                return res.status(500).send('Error reading chart data');
            }
            
            res.setHeader('Content-Type', 'application/json');
            res.send(data);
        });
    } else {
        res.status(404).send('Price chart data not found');
    }
});

app.listen(PORT, () => {
    console.log(`Backtest visualization server running on http://localhost:${PORT}`);
});
