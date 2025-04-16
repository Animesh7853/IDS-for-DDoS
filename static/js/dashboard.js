// WebSocket connection
let socket;
let reconnectInterval;
let charts = {};

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    initCharts();
    setInterval(updateClock, 1000);
});

// Initialize WebSocket connection
function initWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/traffic`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function(e) {
        console.log('WebSocket connection established');
        clearInterval(reconnectInterval);
        document.getElementById('connection-status').textContent = 'Connected';
        document.getElementById('connection-status-indicator').className = 'status-indicator connected';
    };
    
    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
    
    socket.onclose = function(event) {
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status-indicator').className = 'status-indicator disconnected';
        
        // Try to reconnect after 5 seconds
        reconnectInterval = setTimeout(() => {
            console.log('Attempting to reconnect...');
            initWebSocket();
        }, 5000);
    };
    
    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
        document.getElementById('connection-status').textContent = 'Error';
        document.getElementById('connection-status-indicator').className = 'status-indicator error';
    };
}

// Initialize Chart.js charts
function initCharts() {
    // Traffic chart - Requests over time
    const trafficCtx = document.getElementById('traffic-chart').getContext('2d');
    charts.traffic = new Chart(trafficCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Requests',
                data: [],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Requests'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    });
    
    // Attack distribution chart
    const attackDistCtx = document.getElementById('attack-distribution-chart').getContext('2d');
    charts.attackDist = new Chart(attackDistCtx, {
        type: 'pie',
        data: {
            labels: [],
            datasets: [{
                label: 'Attack Types',
                data: [],
                backgroundColor: [
                    'rgba(46, 204, 113, 0.7)',  // BENIGN - Green
                    'rgba(231, 76, 60, 0.7)',   // Red
                    'rgba(241, 196, 15, 0.7)',  // Yellow
                    'rgba(52, 152, 219, 0.7)',  // Blue
                    'rgba(155, 89, 182, 0.7)',  // Purple
                    'rgba(52, 73, 94, 0.7)'     // Dark Blue-Gray
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // Hourly traffic chart
    const hourlyCtx = document.getElementById('hourly-chart').getContext('2d');
    charts.hourly = new Chart(hourlyCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 24}, (_, i) => `${String(i).padStart(2, '0')}:00`),
            datasets: [{
                label: 'Requests per Hour',
                data: Array(24).fill(0),
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Requests'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day'
                    }
                }
            }
        }
    });
}

// Update the dashboard with new data
function updateDashboard(data) {
    // Update summary statistics
    document.getElementById('total-requests').textContent = data.total_requests || 0;
    document.getElementById('avg-confidence').textContent = ((data.average_confidence || 0) * 100).toFixed(2) + '%';
    document.getElementById('avg-response-time').textContent = (data.average_processing_time || 0).toFixed(2) + ' ms';
    
    // Calculate percentage of benign traffic
    const benignCount = data.request_counts_by_result?.BENIGN || 0;
    const totalRequests = data.total_requests || 1;  // Prevent division by zero
    const benignPercent = (benignCount / totalRequests * 100).toFixed(2);
    document.getElementById('benign-percentage').textContent = benignPercent + '%';
    
    // Update attack distribution chart
    if (charts.attackDist && data.request_counts_by_result) {
        const labels = Object.keys(data.request_counts_by_result);
        const values = labels.map(label => data.request_counts_by_result[label]);
        
        charts.attackDist.data.labels = labels;
        charts.attackDist.data.datasets[0].data = values;
        charts.attackDist.update();
    }
    
    // Update hourly traffic chart
    if (charts.hourly && data.requests_per_hour) {
        const hourlyData = Array(24).fill(0);
        
        Object.keys(data.requests_per_hour).forEach(hour => {
            const hourIndex = parseInt(hour);
            if (!isNaN(hourIndex) && hourIndex >= 0 && hourIndex < 24) {
                hourlyData[hourIndex] = data.requests_per_hour[hour];
            }
        });
        
        charts.hourly.data.datasets[0].data = hourlyData;
        charts.hourly.update();
    }
    
    // Update traffic chart (add new data point)
    if (charts.traffic) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        
        // Add new data point for current request rate
        charts.traffic.data.labels.push(timeStr);
        charts.traffic.data.datasets[0].data.push(data.current_request_rate || 0);
        
        // Keep only the last 12 points
        if (charts.traffic.data.labels.length > 12) {
            charts.traffic.data.labels.shift();
            charts.traffic.data.datasets[0].data.shift();
        }
        
        charts.traffic.update();
    }
    
    // Update recent attacks list
    const attacksList = document.getElementById('attack-list');
    if (attacksList && data.recent_attacks) {
        attacksList.innerHTML = '';
        
        if (data.recent_attacks.length === 0) {
            const noAttacksItem = document.createElement('div');
            noAttacksItem.className = 'attack-item';
            noAttacksItem.textContent = 'No recent attacks detected';
            attacksList.appendChild(noAttacksItem);
        } else {
            data.recent_attacks.forEach(attack => {
                const attackItem = document.createElement('div');
                attackItem.className = 'attack-item';
                
                const attackType = document.createElement('div');
                attackType.className = 'attack-type attack';
                attackType.textContent = attack.prediction;
                
                const attackTime = document.createElement('div');
                attackTime.className = 'attack-time';
                attackTime.textContent = new Date(attack.timestamp).toLocaleString();
                
                const attackDetails = document.createElement('div');
                attackDetails.textContent = `From: ${attack.client_ip} (Confidence: ${(attack.confidence * 100).toFixed(2)}%)`;
                
                attackItem.appendChild(attackType);
                attackItem.appendChild(attackTime);
                attackItem.appendChild(attackDetails);
                attacksList.appendChild(attackItem);
            });
        }
    }
    
    // Update alerts
    const alertsContainer = document.getElementById('alerts-container');
    if (alertsContainer && data.alerts) {
        // Clear existing alerts
        alertsContainer.innerHTML = '';
        
        // Add new alerts
        data.alerts.forEach(alert => {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.level}`;
            alertDiv.textContent = alert.message;
            alertsContainer.appendChild(alertDiv);
        });
    }
    
    // Update top source IPs table
    if (data.top_source_ips) {
        const tableBody = document.getElementById('ip-table-body');
        if (tableBody) {
            tableBody.innerHTML = '';
            
            Object.entries(data.top_source_ips)
                .sort((a, b) => b[1] - a[1])
                .forEach(([ip, count]) => {
                    const row = document.createElement('tr');
                    
                    const ipCell = document.createElement('td');
                    ipCell.textContent = ip;
                    
                    const countCell = document.createElement('td');
                    countCell.textContent = count;
                    
                    row.appendChild(ipCell);
                    row.appendChild(countCell);
                    tableBody.appendChild(row);
                });
        }
    }
}

// Update the clock
function updateClock() {
    const clockEl = document.getElementById('current-time');
    if (clockEl) {
        const now = new Date();
        clockEl.textContent = now.toLocaleString();
    }
}

// Send a ping to keep the WebSocket alive
setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send('ping');
    }
}, 30000); // Every 30 seconds
