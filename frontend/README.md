# Interactive Gantt Chart Frontend

A modern, interactive web-based Gantt chart visualization for production scheduling results, built with TypeScript/JavaScript and D3.js.

## Features

### 📊 Dual Gantt Charts
- **Order-Level Gantt**: Shows overall planned start and end times for each order
- **Machine-Level Gantt**: Displays individual operations scheduled on each machine with color-coded orders

### 🎛️ Interactive Controls
- **Zoom Control**: Adjustable zoom level (0.5x to 3.0x) with slider
- **Pan & Zoom**: Mouse-driven pan and zoom functionality
- **Order Filtering**: Multi-select dropdown to filter specific orders
- **Fit to Screen**: Auto-adjust zoom to fit all data optimally
- **Reset View**: Quick reset to default zoom and position

### ✨ Advanced Features
- **Responsive Tooltips**: Hover over tasks to see detailed information
- **Color-Coded Legend**: Each order has a unique color for easy identification
- **Real-time Statistics**: Live updates of total orders, machines, schedule duration, and utilization
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with smooth animations

### 📈 Statistics Dashboard
- Total number of orders and machines
- Overall schedule duration
- Machine utilization percentage
- Real-time updates based on filtered data

## Quick Start

### Option 1: Using Python Server (Recommended)
```bash
# Navigate to the frontend directory
cd frontend

# Start the server
python3 server.py

# Open your browser to http://localhost:8000
```

### Option 2: Using Node.js (if available)
```bash
# Install a simple HTTP server
npm install -g http-server

# Navigate to frontend directory and start server
cd frontend
http-server -p 8000 -c-1

# Open your browser to http://localhost:8000
```

### Option 3: Using any other HTTP server
The application works with any HTTP server. Just serve the `frontend` directory and open `index.html`.

## File Structure

```
frontend/
├── index.html              # Main HTML structure
├── styles.css              # Modern CSS styling with animations
├── gantt-chart.js          # Main JavaScript application logic
├── server.py               # Simple Python HTTP server
├── output_est_eet_weighted_input.json  # Scheduling data
└── README.md               # This file
```

## Data Format

The application expects JSON data in the following format:

```json
{
  "workTasks": [
    {
      "id": 1,
      "plannedStart": 1.0,
      "plannedEnd": 6.0
    }
  ],
  "machines": [
    {
      "id": "1",
      "workTasks": [
        {
          "id": 1,
          "plannedStart": 2.0,
          "plannedEnd": 3.0
        }
      ]
    }
  ]
}
```

## Technical Details

### Technologies Used
- **HTML5**: Semantic structure
- **CSS3**: Modern styling with flexbox/grid, animations, and responsive design
- **JavaScript (ES6+)**: Application logic with TypeScript-style JSDoc annotations
- **D3.js v7**: Data visualization and SVG manipulation
- **Python 3**: Simple HTTP server for local development

### Browser Compatibility
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

### Performance Features
- Efficient SVG rendering with D3.js
- Optimized data filtering and processing
- Smooth animations with CSS transitions
- Responsive design with CSS Grid and Flexbox

## Customization

### Colors
The application uses a predefined color scheme from D3's `schemeCategory10`. To customize colors, modify the `colorScale` in `gantt-chart.js`:

```javascript
this.colorScale = d3.scaleOrdinal(['#your', '#custom', '#colors']);
```

### Styling
All visual styling is contained in `styles.css`. The design uses CSS custom properties for easy theming:

```css
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --background-color: #f5f7fa;
}
```

### Data Source
To use different data, replace `output_est_eet_weighted_input.json` or modify the fetch URL in the `loadData()` method.

## Troubleshooting

### Common Issues

1. **Data not loading**: Ensure the JSON file is in the same directory and the server is running
2. **CORS errors**: Use the provided Python server or another HTTP server (don't open the HTML file directly)
3. **Charts not rendering**: Check browser console for JavaScript errors
4. **Responsive issues**: Clear browser cache and ensure viewport meta tag is present

### Debug Mode
Open browser developer tools (F12) to see console logs and debug information.

## Future Enhancements

- Export functionality (PNG, SVG, PDF)
- Timeline scrubbing and animation
- Advanced filtering options
- Multiple scheduling algorithm comparison
- Real-time data updates via WebSocket
- Drag-and-drop task rescheduling

## License

This project is part of the APS_RPC_demo system. Please refer to the main project license.

## Support

For issues or questions, please refer to the main project documentation or create an issue in the project repository.