'use strict'
//import * as data from '../../weather_data_arkona/daily_data/daily_weather_data.json'

$(document).ready(function () {
    const inputLayerNeurons = 20;
    const inputLayerShape = 20;

    const rnn_input_layer_features = 20;
    const rnn_input_layer_timesteps = inputLayerNeurons / rnn_input_layer_features;

    const rnn_input_shape  = [rnn_input_layer_timesteps, rnn_input_layer_features];
    const rnn_output_neurons = 1;

    const rnn_batch_size = 32;

    const output_layer_shape = rnn_output_neurons;
    const output_layer_neurons = 1;
    const numberLayers = 5;



    async function fetchData() {
        let trainData = []
        await $.getJSON("../weather_data_arkona/daily_data/daily_weather_data.json", (data) => {
            trainData = data;
        })
        return trainData
    }
    
    function toDate(intDate) {
       
        let dateString = intDate.toString()
        const dateValues = {
            year: dateString.substring(0,4),
            month: dateString.substring(4,6),
            day: dateString.substring(6,8)
        }
        //const date = new Date(dateValues.year, dateValues.month, dateValues.day).toLocaleDateString('de-DE')
        const formatedDateString = dateValues.year + '-' + dateValues.month + '-' + dateValues.day
        //console.log(formatedDateString)
        return formatedDateString
    }

    async function getData() {
        const weatherData =  await fetchData()
        const cleaned = weatherData.map(d => ({
             date: toDate(d.MESS_DATUM),
             temp_avg: isString(d.TMK) ? parseFloat(d.TMK.replace(',', '.')) : d.TMK,
        }))
        .filter(d => (d.date != null && d.temp_avg != null))
        .reverse()
        for(let i = cleaned.length; i != 1000; i--){
            cleaned.pop()
        }
        //console.log(cleaned)
        return cleaned
    }

    function createModel(data) {
        const model = tf.sequential()

        model.add(tf.layers.dense({units: 20, inputShape:[20]}))
        model.add(tf.layers.reshape({targetShape: [20,1]}))
        let lstmCells = [];
        for(let i = 0; i < numberLayers; i++){
            lstmCells.push(tf.layers.lstmCell({units: 1}))
        }
        //model.add(tf.layers.dense({units: 20, inputShape: [20]}))
        

         model.add(tf.layers.rnn({
             cell: lstmCells,
             inputShape: [20,1],
             returnSequences: false
         }))
        //model.add(tf.layers.lstm({units: 1, inputShape:[20,1]}))  // das alleine funkt halbwegs
        //model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

        model.add(tf.layers.dense({units: 1, inputShape: [1]}))

        return model
    }

    function isString(x) {
        return Object.prototype.toString.call(x) === "[object String]"
    }

    function convertToTensors(data) {
        let x = []
        let y = []
        //console.log(data)
        let tempData = data.map(d => ({
            temp: isString(d.temp_avg) ? parseFloat(d.temp_avg.replace(',','.')) : d.temp_avg,
        }))

        for(let i = 0; i < tempData.length - 20; i++){
            let backvals = []

            for(let j = i + 1; j < i + 21; j++){
                backvals.push(tempData[j])  
            }
            y.push(tempData[i])
            x.push(backvals.reverse())
        }
        
        
         return tf.tidy(() => {

            const inputs = x.map(d => {
                let y = []
                for(let i = 0; i < d.length; i++){
                     y.push(d[i].temp)
                }
                return y
            })
            const outputs = y.map(d => d.temp)
            // console.log(inputs)
            // console.log(outputs)
            const inputTensor = tf.tensor2d(inputs, [inputs.length, inputs[0].length]).div(tf.scalar(10)) //.reshape([inputs.length, inputs[0].length, 1])
            const outputTensor = tf.tensor2d(outputs, [outputs.length, 1]).div(tf.scalar(10)).reshape([outputs.length,1])
           
            //const inputTensor1 = tf.tensor2d(inputs, [inputs.length, inputs[0].length]).
            //console.log(x)
            //console.log(inputs)
            console.log(inputTensor.print())
            console.log(outputTensor.print())
            return {
                inputs: inputTensor,
                outputs: outputTensor,
            }
         })
    }

    async function trainModel(model, inputs, outputs) {
        
        model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse','accuracy'],
        })
        // console.log(inputs.print())
        // console.log(outputs.print())
        const batchSize = 40
        const epochs = 20

        return await model.fit(inputs, outputs, {
            batchSize,
            epochs,
            callbacks: tfvis.show.fitCallbacks(
                {name: 'Training Performance'},
                ['loss','mse','acc'],
                {height: 200, callbacks: ['onEpochEnd']}
            )
        })
    }

    async function modelPredict(model, inputTensor) {
        const outputs = await model.predict(inputTensor).mul(10)  //div(tf.scalar(10)) bei neuen predictions( inputwerten)

            return Array.from(outputs.dataSync())
    }
    function showPlot(data, prediction) {

        
        const trace1 = {
            type: "scatter",
            mode: "lines",
            name: "Temp",
            x: data.map(d => {
                return d.date
            }),
            y: data.map(d => {
                return d.temp_avg
            }),
            line: {color: '#17BECF'}
        }
        const trace2 = {
            type: "scatter",
            mode: "lines",
            name: "Temp_pred",
            x: data.map(d => {
                return d.date
            }),
            y: prediction , 
            line: {color: '#610B0B'}         
        }
        const dataPlot = [trace1, trace2]
        const layout = {
            title: 'Basic Time Series'
        }
        Plotly.newPlot('myDiv', dataPlot, layout)
    }
    async function run() {
        const data = await getData()
        // const xs = tf.tidy(() => {
        //     const xs = tf.scalar(5)

        //     return xs.dataSync()
        // }) 
        const values = data.map(d => ({
            x: d.date,
            y: d.temp_avg,
        }))

        const model = createModel(data)
        tfvis.show.modelSummary({name: 'Model Summary'}, model)

        const tensorData = convertToTensors(data)
        const {inputs, outputs} = tensorData
        // console.log(inputs.shape)
        // inputs.print()
        // console.log(outputs.shape)
        // outputs.print()
        await trainModel(model, inputs, outputs)
        console.log('test')

        const prediction = await modelPredict(model, inputs)

        console.log(prediction)
        // console.log(xs)
        //convertToTensors(data)
        // tfvis.render.scatterplot(
        //     {name: 'Average Temperature Arkona'},
        //     {values},
        //     {
        //         xLabel: 'Date',
        //         yLabel: 'Temp',
        //         zoomToFit: true,
        //         height: 300
        //     }
        // )

        showPlot(data, prediction)
    }
    run()
});