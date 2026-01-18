// CodeBuggy Web App JavaScript

// Store examples data
let examplesData = {};

document.addEventListener('DOMContentLoaded', function() {
    const buggyCodeInput = document.getElementById('buggy-code');
    const fixedCodeInput = document.getElementById('fixed-code');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const exampleSelect = document.getElementById('example-select');
    const resultsSection = document.getElementById('results');
    const errorSection = document.getElementById('error');

    // Load examples data
    loadExamples();

    // Example select handler
    exampleSelect.addEventListener('change', function() {
        const exampleId = this.value;
        if (exampleId && examplesData[exampleId]) {
            const example = examplesData[exampleId];
            buggyCodeInput.value = example.buggy;
            fixedCodeInput.value = example.fixed;
            hideResults();
            hideError();
        }
    });

    // Predict button handler
    predictBtn.addEventListener('click', async function() {
        const buggyCode = buggyCodeInput.value.trim();
        const fixedCode = fixedCodeInput.value.trim();

        if (!buggyCode || !fixedCode) {
            showError('Please provide both buggy and fixed code');
            return;
        }

        // Show loading state
        predictBtn.disabled = true;
        document.querySelector('.btn-text').style.display = 'none';
        document.querySelector('.btn-loading').style.display = 'inline';
        hideResults();
        hideError();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    buggy_code: buggyCode,
                    fixed_code: fixedCode,
                }),
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Unknown error occurred');
            }
        } catch (error) {
            showError('Network error: ' + error.message + '\n\nMake sure the server is running and MLflow is accessible.');
        } finally {
            // Reset button state
            predictBtn.disabled = false;
            document.querySelector('.btn-text').style.display = 'inline';
            document.querySelector('.btn-loading').style.display = 'none';
        }
    });

    // Clear button handler
    clearBtn.addEventListener('click', function() {
        buggyCodeInput.value = '';
        fixedCodeInput.value = '';
        exampleSelect.value = '';
        hideResults();
        hideError();
    });

    async function loadExamples() {
        try {
            const response = await fetch('/api/examples');
            const data = await response.json();
            
            // Store examples in a map for easy lookup
            data.examples.forEach(example => {
                examplesData[example.id] = example;
            });
        } catch (error) {
            console.error('Failed to load examples:', error);
        }
    }

    function displayResults(data) {
        // Show results section
        resultsSection.style.display = 'block';

        // Display graph probability
        const graphProb = data.graph_probability;
        const graphProbPercent = (graphProb * 100).toFixed(2);
        document.getElementById('graph-prob-fill').style.width = graphProbPercent + '%';
        document.getElementById('graph-prob-text').textContent = graphProbPercent + '%';

        // Display stats
        document.getElementById('num-nodes').textContent = data.num_nodes;
        document.getElementById('num-edges').textContent = data.num_edges;

        // Display node predictions
        displayNodePredictions(data.node_predictions);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function displayNodePredictions(predictions) {
        const tableDiv = document.getElementById('predictions-table');
        tableDiv.innerHTML = '';

        // Create header
        const header = document.createElement('div');
        header.className = 'prediction-row prediction-header';
        header.innerHTML = `
            <div>Rank</div>
            <div>Probability</div>
            <div>Node Type</div>
            <div>Label</div>
            <div>Location</div>
        `;
        tableDiv.appendChild(header);

        // Create rows
        predictions.forEach((pred, index) => {
            const row = document.createElement('div');
            row.className = 'prediction-row';

            const prob = pred.probability;
            const probPercent = (prob * 100).toFixed(2);
            let probClass = 'prob-low';
            if (prob > 0.7) probClass = 'prob-high';
            else if (prob > 0.4) probClass = 'prob-medium';

            const location = pred.line && pred.col 
                ? `L${pred.line}:C${pred.col}` 
                : '-';

            row.innerHTML = `
                <div class="rank">#${index + 1}</div>
                <div><span class="prob-badge ${probClass}">${probPercent}%</span></div>
                <div class="node-type">${pred.node_type}</div>
                <div class="node-label">${pred.label || '-'}</div>
                <div class="location">${location}</div>
            `;

            tableDiv.appendChild(row);
        });
    }

    function showError(message) {
        errorSection.style.display = 'block';
        document.getElementById('error-message').textContent = message;
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    function hideError() {
        errorSection.style.display = 'none';
    }
});
