require('dotenv').config();  // Make sure dotenv is loaded at the very top
const mongoose = require('mongoose');

// Temporarily hardcode the MongoDB URI for testing
const uri = 'mongodb+srv://arkadeepchakra2003:ITSPROJECT@cluster0.fqcyf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';

const connectDB = async () => {
    try {
        await mongoose.connect(uri, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });
        console.log('MongoDB connected successfully');
    } catch (err) {
        console.error('MongoDB connection failed:', err.message);
        process.exit(1);
    }
};

module.exports = connectDB;


// .env k tante parchilo na tai alada kora erokom vabe direct kora