'use strict';

const tfjs = require('@tensorflow/tfjs')
const express = require('express')
const app = express()
const fs = require('fs');
const port = 3000

// const data = require('../weather_data_arkona/daily_data/daily_weather_data.json')
//   app.get('/', (req, res) => {
//       res.sendFile(data)
//   })




module.exports = app