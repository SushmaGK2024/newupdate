// server/server.js
const express = require('express');
const cors = require('cors');
const app = express();
const port = 3001;

// Add this line to parse JSON bodies
app.use(express.json());

// Set up CORS to allow requests from the Netlify domain
app.use(cors({
  origin: 'https://majortrail1.netlify.app',
  credentials: true
}));

// Set additional CORS headers
app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "https://majortrail1.netlify.app");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

// Define your routes...
const usersRoutes = require('./routes/users.js');
const experiencesRoutes = require('./routes/experiences.js');
const placementsRoutes = require('./routes/placements.js');
const extractQuestions = require('./routes/extract_questions.js');
app.use('/users', usersRoutes);
app.use('/experiences', experiencesRoutes);
app.use('/placements', placementsRoutes);
app.use('/extract-questions', extractQuestions);

// Define root path route
app.get('/', (req, res) => {
  res.send('Hello, this is the root path!');
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
