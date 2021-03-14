'use strict'
//import * as data from '../../weather_data_arkona/daily_data/daily_weather_data.json'

$(document).ready(function () {    

    const numberLayers = 1;

    //User Input aus Feldern bekommen
    async function getUserInput() {
        let optimizer = await $('#optimizer').val()
        let epochs = await $('#epochs').val()
        let learningRate = await $('#learningRate').val()
        let date = await $('#datePicker').val()

        return {
            'optimizer' : optimizer,
            'epochs' : parseInt(epochs),
            'learningRate' : parseFloat(learningRate),
            'date' : date
            }
    }

    //Lade die Wetterdaten aus der JSON Datei und gebe sie zurück
    async function fetchData() {
        let trainData = []
        await $.getJSON("../weather_data_arkona/daily_data/daily_weather_data.json", (data) => {
            trainData = data;
        })
        return trainData
    }
    
    //Wandelt die die daten, welche als 8-stelliger Integer gegeben sind in einen String um, Format: YYYY-MM-DD
    function toDateString(intDate) {
       
        let dateString = intDate.toString()
        const dateValues = {
            year: dateString.substring(0,4),
            month: dateString.substring(4,6),
            day: dateString.substring(6,8)
        }
        const formatedDateString = dateValues.year + '-' + dateValues.month + '-' + dateValues.day
        return formatedDateString
    }

    // Erstellt einen Array, welcher die aktuellsten 1000 durschnittlichen Temperaturen enthält. Diese werden hierbei noch aufgrund
    // der Formatierung umgewandelt/geändert
    async function getData() {
        const weatherData =  await fetchData()
        const cleaned = weatherData.map(d => ({
             date: toDateString(d.MESS_DATUM),
             temp_avg: isString(d.TMK) ? parseFloat(d.TMK.replace(',', '.')) : d.TMK,
        }))
        .filter(d => (d.date != null && d.temp_avg != null))
        .reverse()
        for(let i = cleaned.length; i != 1000; i--){
            cleaned.pop()
        }

        return cleaned
    }

    //Erstellt das RNN Model mit den passenden Shapes und LSTM Zellen
    function createModel() {
        const model = tf.sequential()

        model.add(tf.layers.dense({units: 20, inputShape:[20]}))
        model.add(tf.layers.reshape({targetShape: [20,1]}))
        let lstmCells = [];
        for(let i = 0; i < numberLayers; i++){
            lstmCells.push(tf.layers.lstmCell({units: 1}))
        }

        // Erstellung des RNN
         model.add(tf.layers.rnn({
             cell: lstmCells,
             inputShape: [20,1],
             returnSequences: false
         }))


        model.add(tf.layers.dense({units: 1, inputShape: [1]}))

        return model
    }

    // Prüft ob es sich um einen String handelt
    function isString(x) {
        return Object.prototype.toString.call(x) === "[object String]"
    }

    //Umwandlung der Daten in die jeweiligen input und output Tensoren, welche für das Training dienen
    //Hierbei weden die letzten 20 Werte entfernt, da immer 20 Werte zur Vorhersage des 21sten Wertes benutzt werden
    //Division der Tensoren mit 10, damit Werte kleiner sind und sich so besser für das Training des Modelss eignen(geringerer Abstand zwischen Werten)
    //Inputs werden in Reihenfolge getauscht, da der aktuellster Wert für die Vorhersage an erster Stelle im Array steht
    function convertToTensors(data) {
        let x = []
        let y = []

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

            const inputTensor = tf.tensor2d(inputs, [inputs.length, inputs[0].length]).div(tf.scalar(10))
            const outputTensor = tf.tensor2d(outputs, [outputs.length, 1]).div(tf.scalar(10)).reshape([outputs.length,1])
           
            return {
                inputs: inputTensor,
                outputs: outputTensor,
            }
         })
    }

    //Trainieren des Models mit User Werten sowie den Inputs und Outputs, sowie Darstellung der Epochendurchläufe mit Metriken im Debug Fenster
    // zur besseren Verfolgung
    async function trainModel(model, inputs, outputs) {
        let userInput = await getUserInput()
        
        let getOptimizer = (userInput) => {
            let learningRate = userInput.learningRate
            switch (userInput.optimizer) {
                case "RMSProp":
                    return tf.train.rmsprop(learningRate)
            
                case "Adam":
                    return tf.train.adam(learningRate)
                
                case "Stochastic Gradient Decent":
                    return tf.train.sgd(learningRate)
            }
        }
        model.compile({
            optimizer: getOptimizer(userInput),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse','accuracy'],
        })
        const batchSize = 40
        const epochs = userInput.epochs

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

    //Vorhersage auf Basis des Models und der Trainingsdaten
    // multiplikation mit 10 um wieder richtige Skalierung zu erhalten
    async function modelPredict(model, inputTensor) {
        const outputs = await model.predict(inputTensor).mul(10)  //div(tf.scalar(10)) bei neuen predictions( inputwerten)
        console.log(outputs)
            return Array.from(outputs.dataSync())
    }

    //Vorhersage der komplett neuen Werte auf Basis der vorherigen Predictions
    //Werte werden in der Schleife immer wieder angepasst, da Input immer die jeweils neue Prediction enthalten muss um eine neue Vorhersage zu treffen
    async function modelPredictNew(model, predictedValues) {
        let userInput =  await getUserInput()

        let userPickedDate = new Date(userInput.date + " 12:00")
        let minDate = new Date("2019-01-01 12:00")
        let dateIncrement = minDate

        let newDates = []
        let outputs = []

        let predictedValuesArr = predictedValues.slice(0,20)

        while(dateIncrement <= userPickedDate) {
            //console.log('Hello')
            let inputs = [predictedValuesArr]
            newDates.push(dateIncrement.toJSON().split("T")[0])

            let output = await model.predict(tf.tensor2d(inputs, [inputs.length, inputs[0].length]).div(tf.scalar(10))).mul(tf.scalar(10)).dataSync()

            outputs.push(output[0])
            predictedValuesArr.unshift(output[0])
            predictedValuesArr.pop()


            dateIncrement.setDate(dateIncrement.getDate() + 1)
        }

        return {
            outputs: outputs,
            dates: newDates
        }
    }

    //Darstellung der Ausgangsdaten sowie der Vorhergesagten Daten im Plot. Auf der x-Achse das Datum, auf Y-Achse die Temperatur.
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

    //Update des Graphen mit den neuen Vorhersagen, eigene Methode da erst Idee auf basis eines neuen Buttons zu nutzen
    function updatePlot(predictionNew) {
        
        const trace3 = {
            type: "scatter",
            mode: predictionNew.outputs.length != 1 ? "lines" : "marker",
            line: {
                color: '#FFA500'
            },
            name: "Temp_pred_new",
            x: predictionNew.dates,
            y: predictionNew.outputs
        }

        Plotly.addTraces('myDiv', trace3)
    }

    //Main Function für den gesamten Durchlauf
    async function run() {
        const data = await getData()
        
        const values = data.map(d => ({
            x: d.date,
            y: d.temp_avg,
        }))

        const model = createModel()
        tfvis.show.modelSummary({name: 'Model Summary'}, model)

        const tensorData = convertToTensors(data)
        const {inputs, outputs} = tensorData

        await trainModel(model, inputs, outputs)


        const prediction = await modelPredict(model, inputs)

        const predictionNew = await modelPredictNew(model, prediction)

        showPlot(data, prediction)

        updatePlot(predictionNew)
    }
    
    //Aufüren des gesamten Vorgang nach Klick auf Button
    $("#actiontrain").on("click", () => {
        run();
    })

    // Begrenzung der Auswahl des Datums
    $('#datePicker').prop('min', () => {
        return new Date("2019-01-01 12:00").toJSON().split("T")[0]
    }).prop('max', () => {
        return new Date("2019-12-31 12:00").toJSON().split("T")[0]
    })
});