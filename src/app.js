'use strict';

const tfjs = require('@tensorflow/tfjs')
const express = require('express')
const app = express()
const fs = require('fs');
const { checkServerIdentity } = require('tls');
const port = 3000

// app.get('/', (req, res) => {
//     res.send('Hello World!')
// })



module.exports = app