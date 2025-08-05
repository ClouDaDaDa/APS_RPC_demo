// TypeScript-style interfaces (using JSDoc for type checking)

/**
 * @typedef {Object} WorkTask
 * @property {number} id - Task/Order ID
 * @property {number} plannedStart - Planned start time
 * @property {number} plannedEnd - Planned end time
 */

/**
 * @typedef {Object} Machine
 * @property {string} id - Machine ID
 * @property {WorkTask[]} workTasks - Array of work tasks assigned to this machine
 */

/**
 * @typedef {Object} ScheduleData
 * @property {WorkTask[]} workTasks - Array of overall work tasks (orders)
 * @property {Machine[]} machines - Array of machines with their assigned tasks
 */

class GanttChart {
    constructor() {
        this.data = null;
        this.orderColors = {};
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        this.zoomLevel = 1;
        this.selectedOrders = new Set();
        
        this.margins = {
            top: 20,
            right: 80,
            bottom: 40,
            left: 100
        };
        
        this.init();
    }

    async init() {
        try {
            await this.loadData();
            this.setupEventListeners();
            this.renderCharts();
            this.updateStatistics();
        } catch (error) {
            console.error('Failed to initialize Gantt Chart:', error);
            this.showError('Failed to load scheduling data. Please check the data file.');
        }
    }

    async loadData() {
        try {
            const response = await fetch('./output_EST_SPT_weighted_input_test_generated.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.data = await response.json();
            this.processData();
        } catch (error) {
            console.error('Error loading data:', error);
            throw error;
        }
    }

    processData() {
        if (!this.data || !this.data.workTasks || !this.data.machines) {
            throw new Error('Invalid data format');
        }

        // Filter out invalid tasks (plannedStart or plannedEnd = -1)
        this.data.machines.forEach(machine => {
            machine.workTasks = machine.workTasks.filter(task => 
                task.plannedStart >= 0 && task.plannedEnd >= 0
            );
        });

        // Generate color mapping for orders
        const orderIds = [...new Set(this.data.workTasks.map(task => task.id))].sort((a, b) => a - b);
        orderIds.forEach((id, index) => {
            this.orderColors[id] = this.colorScale(index);
        });

        // Initialize order filter
        this.setupOrderFilter(orderIds);
        
        // Initialize selected orders (all selected by default)
        this.selectedOrders = new Set(orderIds);
    }

    setupOrderFilter(orderIds) {
        const select = document.getElementById('order-filter');
        select.innerHTML = '<option value="all" selected>All Orders</option>';
        
        orderIds.forEach(id => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `Order ${id}`;
            option.selected = true;
            select.appendChild(option);
        });
    }

    setupEventListeners() {
        // Zoom control
        const zoomSlider = document.getElementById('zoom-slider');
        const zoomValue = document.getElementById('zoom-value');
        
        zoomSlider.addEventListener('input', (e) => {
            this.zoomLevel = parseFloat(e.target.value);
            zoomValue.textContent = `${this.zoomLevel.toFixed(1)}x`;
            this.renderCharts();
        });

        // Reset zoom
        document.getElementById('reset-zoom').addEventListener('click', () => {
            this.zoomLevel = 1;
            zoomSlider.value = 1;
            zoomValue.textContent = '1.0x';
            this.renderCharts();
        });

        // Fit to screen
        document.getElementById('fit-to-screen').addEventListener('click', () => {
            this.fitToScreen();
        });

        // Order filter
        document.getElementById('order-filter').addEventListener('change', (e) => {
            this.updateOrderFilter(e.target);
        });

        // Window resize
        window.addEventListener('resize', () => {
            setTimeout(() => this.renderCharts(), 100);
        });
    }

    updateOrderFilter(selectElement) {
        const selectedOptions = Array.from(selectElement.selectedOptions);
        const selectedValues = selectedOptions.map(option => option.value);
        
        if (selectedValues.includes('all')) {
            // Select all orders
            this.selectedOrders = new Set(Object.keys(this.orderColors).map(Number));
            Array.from(selectElement.options).forEach(option => {
                if (option.value !== 'all') option.selected = true;
            });
        } else {
            this.selectedOrders = new Set(selectedValues.map(Number));
        }
        
        this.renderCharts();
    }

    fitToScreen() {
        const maxTime = Math.max(
            ...this.data.workTasks.map(task => task.plannedEnd),
            ...this.data.machines.flatMap(machine => 
                machine.workTasks.map(task => task.plannedEnd)
            )
        );
        
        const containerWidth = document.querySelector('.gantt-chart').clientWidth;
        const availableWidth = containerWidth - this.margins.left - this.margins.right;
        const optimalZoom = Math.min(3, availableWidth / (maxTime * 30)); // 30px per time unit as baseline
        
        this.zoomLevel = Math.max(0.5, optimalZoom);
        document.getElementById('zoom-slider').value = this.zoomLevel;
        document.getElementById('zoom-value').textContent = `${this.zoomLevel.toFixed(1)}x`;
        this.renderCharts();
    }

    renderCharts() {
        this.renderOrderGantt();
        this.renderMachineGantt();
    }

    renderOrderGantt() {
        const container = d3.select('#order-gantt');
        container.selectAll('*').remove();

        const filteredTasks = this.data.workTasks.filter(task => 
            this.selectedOrders.has(task.id)
        );

        if (filteredTasks.length === 0) {
            container.append('div')
                .attr('class', 'loading')
                .text('No orders selected');
            return;
        }

        const containerRect = container.node().getBoundingClientRect();
        const width = containerRect.width;
        const height = Math.max(400, filteredTasks.length * 50 + this.margins.top + this.margins.bottom);

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const chartWidth = width - this.margins.left - this.margins.right;
        const chartHeight = height - this.margins.top - this.margins.bottom;

        const maxTime = Math.max(...filteredTasks.map(task => task.plannedEnd));
        const minTime = Math.min(...filteredTasks.map(task => task.plannedStart));

        const xScale = d3.scaleLinear()
            .domain([minTime, maxTime])
            .range([0, chartWidth * this.zoomLevel]);

        const yScale = d3.scaleBand()
            .domain(filteredTasks.map(task => task.id).sort((a, b) => a - b))
            .range([0, chartHeight])
            .padding(0.2);

        const g = svg.append('g')
            .attr('transform', `translate(${this.margins.left},${this.margins.top})`);

        // Add grid
        this.addGrid(g, xScale, yScale, chartHeight);

        // Add axes
        this.addAxes(g, xScale, yScale, chartHeight, 'Order');

        // Add task bars
        const bars = g.selectAll('.task-bar')
            .data(filteredTasks)
            .enter().append('rect')
            .attr('class', 'task-bar')
            .attr('x', d => xScale(d.plannedStart))
            .attr('y', d => yScale(d.id))
            .attr('width', d => xScale(d.plannedEnd) - xScale(d.plannedStart))
            .attr('height', yScale.bandwidth())
            .attr('fill', d => this.orderColors[d.id])
            .attr('rx', 4)
            .attr('ry', 4);

        // Add task labels
        g.selectAll('.task-text')
            .data(filteredTasks)
            .enter().append('text')
            .attr('class', 'task-text')
            .attr('x', d => xScale(d.plannedStart) + (xScale(d.plannedEnd) - xScale(d.plannedStart)) / 2)
            .attr('y', d => yScale(d.id) + yScale.bandwidth() / 2)
            .text(d => `${d.plannedStart.toFixed(1)}-${d.plannedEnd.toFixed(1)}`)
            .style('font-size', Math.min(12, yScale.bandwidth() * 0.3) + 'px');

        // Add tooltips
        this.addTooltips(bars, (d) => ({
            title: `Order ${d.id}`,
            items: [
                { label: 'Start Time', value: d.plannedStart.toFixed(2) },
                { label: 'End Time', value: d.plannedEnd.toFixed(2) },
                { label: 'Duration', value: (d.plannedEnd - d.plannedStart).toFixed(2) }
            ]
        }));

        // Add zoom and pan
        this.addZoomPan(svg, g, xScale, chartWidth);
    }

    renderMachineGantt() {
        const container = d3.select('#machine-gantt');
        container.selectAll('*').remove();

        const filteredMachines = this.data.machines.map(machine => ({
            ...machine,
            workTasks: machine.workTasks.filter(task => this.selectedOrders.has(task.id))
        })).filter(machine => machine.workTasks.length > 0);

        if (filteredMachines.length === 0) {
            container.append('div')
                .attr('class', 'loading')
                .text('No data for selected orders');
            return;
        }

        const containerRect = container.node().getBoundingClientRect();
        const width = containerRect.width;
        const height = Math.max(400, filteredMachines.length * 60 + this.margins.top + this.margins.bottom);

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const chartWidth = width - this.margins.left - this.margins.right;
        const chartHeight = height - this.margins.top - this.margins.bottom;

        const allTasks = filteredMachines.flatMap(machine => machine.workTasks);
        const maxTime = Math.max(...allTasks.map(task => task.plannedEnd));
        const minTime = Math.min(...allTasks.map(task => task.plannedStart));

        const xScale = d3.scaleLinear()
            .domain([minTime, maxTime])
            .range([0, chartWidth * this.zoomLevel]);

        const yScale = d3.scaleBand()
            .domain(filteredMachines.map(machine => machine.id).sort((a, b) => parseInt(a) - parseInt(b)))
            .range([0, chartHeight])
            .padding(0.2);

        const g = svg.append('g')
            .attr('transform', `translate(${this.margins.left},${this.margins.top})`);

        // Add grid
        this.addGrid(g, xScale, yScale, chartHeight);

        // Add axes
        this.addAxes(g, xScale, yScale, chartHeight, 'Machine');

        // Add task bars for each machine
        filteredMachines.forEach(machine => {
            const machineGroup = g.append('g')
                .attr('class', 'machine-group');

            const bars = machineGroup.selectAll('.task-bar')
                .data(machine.workTasks)
                .enter().append('rect')
                .attr('class', 'task-bar')
                .attr('x', d => xScale(d.plannedStart))
                .attr('y', yScale(machine.id))
                .attr('width', d => xScale(d.plannedEnd) - xScale(d.plannedStart))
                .attr('height', yScale.bandwidth())
                .attr('fill', d => this.orderColors[d.id])
                .attr('rx', 4)
                .attr('ry', 4);

            // Add task labels
            machineGroup.selectAll('.task-text')
                .data(machine.workTasks)
                .enter().append('text')
                .attr('class', 'task-text')
                .attr('x', d => xScale(d.plannedStart) + (xScale(d.plannedEnd) - xScale(d.plannedStart)) / 2)
                .attr('y', d => yScale(machine.id) + yScale.bandwidth() / 2)
                .text(d => `Order ${d.id}`)
                .style('font-size', Math.min(11, yScale.bandwidth() * 0.25) + 'px');

            // Add tooltips
            this.addTooltips(bars, (d) => ({
                title: `Machine ${machine.id} - Order ${d.id}`,
                items: [
                    { label: 'Machine', value: machine.id },
                    { label: 'Order', value: d.id },
                    { label: 'Start Time', value: d.plannedStart.toFixed(2) },
                    { label: 'End Time', value: d.plannedEnd.toFixed(2) },
                    { label: 'Duration', value: (d.plannedEnd - d.plannedStart).toFixed(2) }
                ]
            }));
        });

        // Add zoom and pan
        this.addZoomPan(svg, g, xScale, chartWidth);

        // Update legend
        this.updateLegend();
    }

    addGrid(g, xScale, yScale, chartHeight) {
        // Vertical grid lines
        const xTicks = xScale.ticks(10);
        g.selectAll('.grid-x')
            .data(xTicks)
            .enter().append('line')
            .attr('class', 'grid')
            .attr('x1', d => xScale(d))
            .attr('x2', d => xScale(d))
            .attr('y1', 0)
            .attr('y2', chartHeight);

        // Horizontal grid lines
        g.selectAll('.grid-y')
            .data(yScale.domain())
            .enter().append('line')
            .attr('class', 'grid')
            .attr('x1', 0)
            .attr('x2', xScale.range()[1])
            .attr('y1', d => yScale(d) + yScale.bandwidth())
            .attr('y2', d => yScale(d) + yScale.bandwidth());
    }

    addAxes(g, xScale, yScale, chartHeight, yAxisLabel) {
        // X-axis
        const xAxis = d3.axisBottom(xScale)
            .tickFormat(d => d.toFixed(1));
        
        g.append('g')
            .attr('class', 'axis x-axis')
            .attr('transform', `translate(0,${chartHeight})`)
            .call(xAxis);

        // Y-axis
        const yAxis = d3.axisLeft(yScale)
            .tickFormat(d => `${yAxisLabel} ${d}`);
        
        g.append('g')
            .attr('class', 'axis y-axis')
            .call(yAxis);

        // Axis labels
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - this.margins.left + 20)
            .attr('x', 0 - (chartHeight / 2))
            .style('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '600')
            .style('fill', '#2c3e50')
            .text(yAxisLabel);

        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', `translate(${xScale.range()[1] / 2}, ${chartHeight + 35})`)
            .style('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '600')
            .style('fill', '#2c3e50')
            .text('Time');
    }

    addTooltips(selection, dataFunction) {
        const tooltip = d3.select('#tooltip');

        selection
            .on('mouseover', (event, d) => {
                const data = dataFunction(d);
                
                let tooltipContent = `<div class="tooltip-title">${data.title}</div>`;
                tooltipContent += '<div class="tooltip-content">';
                data.items.forEach(item => {
                    tooltipContent += `
                        <div class="tooltip-item">
                            <span class="tooltip-label">${item.label}:</span>
                            <span class="tooltip-value">${item.value}</span>
                        </div>
                    `;
                });
                tooltipContent += '</div>';

                tooltip
                    .html(tooltipContent)
                    .classed('visible', true)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mousemove', (event) => {
                tooltip
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => {
                tooltip.classed('visible', false);
            });
    }

    addZoomPan(svg, g, xScale, chartWidth) {
        // 只允许横向拖动，不允许缩放
        const zoom = d3.zoom()
            .scaleExtent([1, 1]) // 禁止缩放
            .on('zoom', (event) => {
                const transform = event.transform;
                g.attr('transform', `translate(${this.margins.left + transform.x},${this.margins.top}) scale(1, 1)`);
            });

        svg.call(zoom)
            .on("wheel.zoom", null)      // 禁止滚轮缩放
            .on("dblclick.zoom", null)   // 禁止双击缩放
            .on("touchstart.zoom", null) // 禁止触控缩放
            .on("touchmove.zoom", null)
            .on("touchend.zoom", null);
    }

    updateLegend() {
        const legendContainer = d3.select('#legend');
        legendContainer.selectAll('*').remove();

        const selectedOrderIds = Array.from(this.selectedOrders).sort((a, b) => a - b);
        
        selectedOrderIds.forEach(orderId => {
            const legendItem = legendContainer.append('div')
                .attr('class', 'legend-item');

            legendItem.append('div')
                .attr('class', 'legend-color')
                .style('background-color', this.orderColors[orderId]);

            legendItem.append('span')
                .attr('class', 'legend-text')
                .text(`Order ${orderId}`);
        });
    }

    updateStatistics() {
        if (!this.data) return;

        const totalOrders = this.data.workTasks.length;
        const totalMachines = this.data.machines.length;
        
        const maxTime = Math.max(...this.data.workTasks.map(task => task.plannedEnd));
        const minTime = Math.min(...this.data.workTasks.map(task => task.plannedStart));
        const scheduleDuration = maxTime - minTime;

        // Calculate machine utilization
        let totalMachineTime = 0;
        let totalWorkTime = 0;
        
        this.data.machines.forEach(machine => {
            const validTasks = machine.workTasks.filter(task => 
                task.plannedStart >= 0 && task.plannedEnd >= 0
            );
            const machineWorkTime = validTasks.reduce((sum, task) => 
                sum + (task.plannedEnd - task.plannedStart), 0
            );
            totalWorkTime += machineWorkTime;
            totalMachineTime += scheduleDuration;
        });

        const utilization = totalMachineTime > 0 ? (totalWorkTime / totalMachineTime * 100) : 0;

        // Update DOM
        document.getElementById('total-orders').textContent = totalOrders;
        document.getElementById('total-machines').textContent = totalMachines;
        document.getElementById('schedule-duration').textContent = scheduleDuration.toFixed(1);
        document.getElementById('machine-utilization').textContent = utilization.toFixed(1) + '%';
    }

    showError(message) {
        const containers = ['#order-gantt', '#machine-gantt'];
        containers.forEach(containerId => {
            d3.select(containerId)
                .selectAll('*').remove()
                .append('div')
                .attr('class', 'loading')
                .style('color', '#e74c3c')
                .text(message);
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GanttChart();
});