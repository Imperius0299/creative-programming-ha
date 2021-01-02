const app = require('./app')
port = process.env.PORT || 3000
// port = 3000

//Verbindung zu localhost Server
app.listen(port, () => {
    console.log(`App is listening on port ${port}`)
})