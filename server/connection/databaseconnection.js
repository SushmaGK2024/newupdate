// backend/db/connection.js
require("dotenv").config()
const mysql = require('mysql2');

const db = mysql.createConnection({
  host: process.env.MYSQLHOST,
  user: process.env.MYSQLUSER,
  password:process.env.MYSQLPASSWORD,
  database: process.env.MYSQLDATABASE,
});

// Connect to MySQL
db.connect((err) => {
  if (err) {
    console.error('MySQL connection failed:', err);
  } else {
    console.log('Connected to MySQL database');
  }
});

module.exports = db;
