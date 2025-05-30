var express = require('express');
var router = express.Router();
const passport = require('passport');
const User = require('../models/User_Schema');
const StreamSelection = require('../models/StreamSelection_Schema');
const MCQ = require('../models/MCQ_Schema');
const path = require('path');
const bcrypt = require('bcrypt');

// Middleware to check if the user has already selected a stream
const checkStreamSelected = async (req, res, next) => {
  if (req.isAuthenticated()) {
    try {
      const user = await User.findById(req.user.id);  // Find the user
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      // Check if the user has a stream selection in the StreamSelection model
      const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });
      if (streamSelection) {
        // If a stream is already selected, redirect to the profile
        return res.redirect('/profile');
      }
    } catch (error) {
      console.error('Error in checkStreamSelected middleware:', error);
      return res.status(500).send('Server error');
    }
  } else {
    return res.status(401).json({ message: 'Please log in to access this resource' });
  }
  next();
};

/* GET home page. */
router.get('/', function (req, res, next) {
  res.render('index', { title: 'Express' });
});



// router.get('/forgotpassword', function (req, res, next) {
//   // res.render('certificate', { title: 'Express' });
//   res.render('forgotpassword');
// });

// GET route to render the forgot password page
router.get('/forgotpassword', function (req, res, next) {
  res.render('forgotpassword'); // renders forgotpassword.ejs/pug/etc.
});



// POST route to handle forgot password logic
// const User = require('../models/User'); // Make sure the path is correct

router.post('/forgotpassword', async function (req, res) {
  const { username, email } = req.body;

  try {
    const user = await User.findOne({ username });

    if (!user) {
      return res.json({ status: 'error', message: 'Username not found' });
    }

    if (user.email !== email) {
      return res.json({ status: 'error', message: 'Email does not match the username' });
    }

    // If matched
    return res.json({ status: 'found' });
  } catch (error) {
    console.error('Error in forgotpassword route:', error);
    return res.status(500).json({ status: 'error', message: 'Server error' });
  }
});



// const bcrypt = require('bcrypt');

// POST route to handle password reset
// const bcrypt = require('bcrypt');
// const User = require('../models/User');

router.post('/resetpassword', async (req, res) => {
  const { username, newPassword } = req.body;

  if (!username || !newPassword) {
    return res.status(400).json({ status: 'error', message: 'Missing username or password' });
  }

  try {
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    const user = await User.findOneAndUpdate(
      { username },
      { password: hashedPassword }
    );

    if (!user) {
      return res.status(404).json({ status: 'error', message: 'User not found' });
    }

    res.json({ status: 'success', message: 'Password updated successfully' });
  } catch (error) {
    console.error('Password reset error:', error);
    res.status(500).json({ status: 'error', message: 'Server error' });
  }
});



// router.get('/certificate', function (req, res, next) {
//   // res.render('certificate', { title: 'Express' });
//   res.render('certificate');
// });


// router.get('/certificate', ensureAuthenticated, async (req, res) => {
//   try {
//     const user = await User.findById(req.user.id);
//     if (!user) return res.status(404).send("User not found");

//     const average = req.query.average || 0; // fallback to 0 if not passed
//     res.render('certificate', { user, average });
//   } catch (error) {
//     console.error("Error rendering certificate:", error);
//     res.status(500).send("Server error");
//   }
// });

router.get('/certificate', ensureAuthenticated, async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    if (!user) return res.status(404).send("User not found");

    // Generate certificate ID if it doesn't exist
    if (!user.certificateId) {
      user.certificateId = `CERT-2025-${user._id.toString().slice(-5).toUpperCase()}`;
      await user.save();
    }

    const average = req.query.average || 0;
    res.render('certificate', {
      user,
      average,
      certificateId: user.certificateId
    });
  } catch (error) {
    console.error("Error rendering certificate:", error);
    res.status(500).send("Server error");
  }
});


// router.get('/certificate', ensureAuthenticated, async (req, res) => {
//   const user = await User.findById(req.user.id);
//   const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });

//   // Pass user and stream data to profile view
//   res.render('certificate', { user, streamChoice: streamSelection.stream_choice });

// });

router.get('/suggestion', (req, res) => {
  const data = req.query.data ? JSON.parse(req.query.data) : {};
  res.render('suggestions', { suggestionData: data });
});

// router.get('/suggestion', async (req, res) => {
//   try {
//     const response = await axios.post('http://localhost:5000/course_suggestion', {
//       user_level_sheet: req.query.data ? JSON.parse(req.query.data) : {}
//     });

//     res.json(response.data);  // Send JSON response back to frontend
//   } catch (error) {
//     res.status(500).json({ error: 'Failed to fetch suggestions' });
//   }
// });


router.get('/index', function (req, res, next) {
  res.render('loginlogout', { title: 'Express' });
});

// Register route
// router.post('/register', async (req, res) => {
//   const { username, password } = req.body;
//   try {
//     let user = await User.findOne({ username });
//     if (user) return res.status(400).json({ message: 'Username already exists' });

//     user = new User({ username, password });
//     await user.save();

//     res.status(201).json({ message: 'User registered successfully' });
//   } catch (err) {
//     res.status(500).json({ error: 'Server error' });
//   }
// });

router.post('/register', async (req, res) => {
  const { username, email, password } = req.body;
  try {
    let user = await User.findOne({ username });
    if (user) return res.status(400).json({ message: 'Username already exists' });

    user = new User({ username, email, password });
    await user.save();

    res.status(201).json({ message: 'User registered successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});


// Add Question route
router.post('/addQuestion', async (req, res) => {


  const data = new MCQ({
    question: req.body.question,
    options: req.body.options,
    correctAnswer: req.body.correctAnswer,
    stream: req.body.stream,
    difficulty: req.body.difficulty,

  })

  try {
    const dataToSave = await data.save();
    res.status(200).json(dataToSave)
  }
  catch (error) {
    res.status(400).json({ message: error.message })
  }


});

// Login route with redirection logic
router.post('/login', passport.authenticate('local'), async (req, res) => {
  try {
    const user = await User.findById(req.user.id);  // Find the user by ID

    // Check if the user has a stream selection
    const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });

    if (!streamSelection) {
      // Redirect to stream selection if the stream is not chosen
      return res.redirect('/select-stream');
    } else {

      //stream_choice
      req.session.stream = streamSelection.stream_choice
      console.log("Selected Stream: " + streamSelection.stream_choice)

      // Redirect to profile if the stream is already chosen
      return res.redirect('/profile');
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error during login redirection' });
  }
});

// Logout route
router.get('/logout', (req, res) => {
  req.logout(err => {
    if (err) return res.status(500).json({ error: 'Failed to log out' });
    return res.redirect("/");
  });
});

// Route to render the stream selection page with checkStreamSelected middleware
router.get('/select-stream', ensureAuthenticated, checkStreamSelected, (req, res) => {
  res.sendFile(path.join(__dirname, '../public/html/select-stream.html'));
});

// Select stream route
router.post('/select-stream', async (req, res) => {
  try {
    const userId = req.user._id;
    const { stream_choice } = req.body;

    // Save the stream selection
    const selection = await StreamSelection.create({
      user_id: userId,
      stream_choice: stream_choice
    });

    // Update user’s stream choice
    await User.findByIdAndUpdate(userId, { stream_choice: stream_choice });

    // Send a JSON response instructing the frontend to redirect
    return res.json({ redirect: '/profile' });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'An error occurred while saving stream selection' });
  }
});

// Route to render the update stream page
// router.get('/update-profile', ensureAuthenticated, async (req, res) => {
//   try {
//     const user = await User.findById(req.user.id);
//     const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });

//     // Set currentStream to the selected stream or a default value if none found
//     const currentStream = streamSelection ? streamSelection.stream_choice : '';

//     // Render the update-profile.ejs page and pass currentStream data
//     res.render('update-profile', { currentStream });
//   } catch (error) {
//     console.error('Error fetching current stream:', error);
//     res.status(500).json({ message: 'Server error while fetching stream data' });
//   }
// });

// // Update stream route
// router.put('/update-profile', async (req, res) => {
//   try {
//     if (!req.isAuthenticated()) {
//       // Return error in JSON format if user is not authenticated
//       return res.status(401).json({ error: 'User not authenticated' });
//     }

//     const userId = req.user._id;
//     const { stream_choice } = req.body;

//     // Check if the user has an existing stream selection
//     let existingSelection = await StreamSelection.findOne({ user_id: userId });

//     if (!existingSelection) {
//       // If no stream selection found, create a new selection for the user
//       existingSelection = await StreamSelection.create({
//         user_id: userId,
//         stream_choice: stream_choice
//       });
//       return res.status(200).json({ message: 'Stream selection created', selection: existingSelection });
//     } 

//     // If a stream selection exists, update it with the new stream choice
//     existingSelection.stream_choice = stream_choice;
//     await existingSelection.save();

//     // Send response back with the updated stream selection
//     res.status(200).json({ message: 'Stream selection updated', selection: existingSelection });
//   } catch (error) {
//     console.error(error);
//     // Send server error in JSON format if any issues occur
//     res.status(500).json({ error: 'An error occurred while updating stream selection' });
//   }
// });




// GET: Render update page with current stream, username, and email
router.get('/update-profile', ensureAuthenticated, async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });

    const currentStream = streamSelection ? streamSelection.stream_choice : '';
    const username = user.username || '';
    const email = user.email || '';

    res.render('update-profile', {
      currentStream,
      username,
      email
    });
  } catch (error) {
    console.error('Error fetching user or stream data:', error);
    res.status(500).json({ message: 'Server error while fetching data' });
  }
});


// PUT: Update stream, username, and email
router.put('/update-profile', async (req, res) => {
  try {
    if (!req.isAuthenticated()) {
      return res.status(401).json({ error: 'User not authenticated' });
    }

    const userId = req.user._id;
    const { stream_choice, username, email } = req.body;

    // Update or create stream selection
    let streamSelection = await StreamSelection.findOne({ user_id: userId });
    if (!streamSelection) {
      streamSelection = await StreamSelection.create({
        user_id: userId,
        stream_choice
      });
    } else {
      streamSelection.stream_choice = stream_choice;
      await streamSelection.save();
    }

    // Update user info
    const user = await User.findById(userId);
    if (user) {
      if (username) user.username = username;
      if (email) user.email = email;
      await user.save();
    }

    res.status(200).json({
      message: 'Updated Successfully',
      selection: streamSelection,
      user: {
        username: user.username,
        email: user.email
      }
    });
  } catch (error) {
    console.error('Error! update failed:', error);
    res.status(500).json({ error: 'An error occurred while updating' });
  }
});

// Profile route
router.get('/profile', ensureAuthenticated, async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });

    if (!streamSelection) {
      return res.status(404).json({ message: 'Stream selection not found' });
    }

    // Pass user and stream data to profile view
    res.render('profile', { user, streamChoice: streamSelection.stream_choice });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error while fetching profile data' });
  }
});

// Delete user route
router.delete('/delete-user', ensureAuthenticated, async (req, res) => {
  try {
    const userId = req.user._id;

    // Delete user from the User collection
    await User.findByIdAndDelete(userId);

    // Optionally, delete the user's stream selection if it exists
    await StreamSelection.deleteMany({ user_id: userId });

    // Log out the user after deletion and send a success message
    req.logout(err => {
      if (err) return res.status(500).json({ error: 'Failed to log out' });
      res.status(200).json({ message: 'User deleted successfully and logged out' });
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'An error occurred while deleting the user' });
  }
});


// Middleware to ensure user is authenticated
function ensureAuthenticated(req, res, next) {
  if (req.isAuthenticated()) return next();
  res.status(401).json({ message: 'Please log in to access this resource' });
}

// MCQ Quiz page route
router.get('/mcq-quiz', ensureAuthenticated, async (req, res) => {
  res.render('mcq-test');
});

// API route to get MCQ questions based on user's stream
router.get('/api/mcqs/question', ensureAuthenticated, async (req, res) => {
  console.log('User authenticated:', req.user); // Log user info
  try {
    const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });
    console.log('Stream selection:', streamSelection); // Log stream selection data

    if (!streamSelection) {
      return res.status(404).json({ message: 'Stream not selected' });
    }

    const question = await MCQ.find({ stream: streamSelection.stream_choice })
      .limit(10);
    console.log('Questions:', question); // Log the fetched questions

    if (question.length === 0) {
      return res.status(404).json({ message: 'No questions found for the selected stream' });
    }

    res.json(question);
  } catch (error) {
    console.error('Error fetching questions:', error);
    res.status(500).json({ message: 'Error fetching questions' });
  }
});

// API route to submit answers and get results
router.post('/api/mcqs/submit', ensureAuthenticated, async (req, res) => {
  try {
    const { answers } = req.body; // Array of { questionId, answer }
    console.log('Received answers:', answers);

    if (!answers || !Array.isArray(answers)) {
      return res.status(400).json({ message: 'Invalid data format' });
    }

    let correctCount = 0;
    let total_easy = 0;
    let total_medium = 0;
    let total_hard = 0;
    let total_right_easy = 0;
    let total_right_medium = 0;
    let total_right_hard = 0;
    let stream = req.session.stream

    console.log(296, stream)

    for (const { questionId, answer } of answers) {


      const question = await MCQ.findById(questionId);
      if (!question) {
        console.log(`Question not found: ${questionId}`);
        continue;
      }

      if (question.difficulty === "easy") {
        total_easy = total_easy + 1
      }
      if (question.difficulty === "medium") {
        total_medium = total_medium + 1
      }
      if (question.difficulty === "hard") {
        total_hard = total_hard + 1
      }



      // Compare the submitted answer index with the correctAnswer index
      if (question.correctAnswer === answer) {
        correctCount++;
        if (question.difficulty === "easy") {
          total_right_easy = total_right_easy + 1
        }
        if (question.difficulty === "medium") {
          total_right_medium = total_right_medium + 1 
        }
        if (question.difficulty === "hard") {
          total_right_hard = total_right_hard + 1
        }
      }


      //count wrong answer category wise
    }

    const totalQuestions = answers.length;
    // res.json({
    //   message: `You answered ${correctCount} out of ${totalQuestions} questions correctly.`,
    //   correctCount,
    //   totalQuestions,
    // });

    // res.json({
    //   message: {"totalquestion":totalQuestions, "right_answer": correctCount, "wrong_answer": (totalQuestions - correctCount), "levellist": level_list}
    // })

    res.json({
      message: {
        "total_question": totalQuestions,
        "total_right": correctCount,
        "total_wrong": (totalQuestions - correctCount),
        "Stream": stream,
        "easy": {
          "total_question": total_easy,
          "right": total_right_easy
        },
        "medium": {
          "total_question": total_medium,
          "right": total_right_medium
        },
        "hard": {
          "total_question": total_hard,
          "right": total_right_hard
        }
      }
    })
  } catch (error) {
    console.error('Error processing submission:', error);
    res.status(500).json({ message: 'Error submitting answers' });
  }
});

const streamResources = {
"CSE": [
    { name: "HTML", google: "https://www.w3schools.com/html/", youtube: "https://www.youtube.com/watch?v=pQN-pnXPaVg" },
    { name: "CSS", google: "https://www.w3schools.com/css/", youtube: "https://www.youtube.com/watch?v=1Rs2ND1ryYc" },
    { name: "Python", google: "https://www.python.org/about/gettingstarted/", youtube: "https://www.youtube.com/watch?v=_uQrJ0TkZlc" },
    { name: "Machine Learning", google: "https://www.geeksforgeeks.org/machine-learning", youtube: "https://www.youtube.com/watch?v=GwIo3gDZCVQ" },
    { name: "Artificial Intelligence", google: "https://www.geeksforgeeks.org/artificial-intelligence", youtube: "https://www.youtube.com/watch?v=2ePf9rue1Ao" },
    { name: "Java", google: "https://www.w3schools.com/java/", youtube: "https://www.youtube.com/watch?v=grEKMHGYyns" },
    { name: "Operating Systems", google: "https://www.geeksforgeeks.org/operating-systems", youtube: "https://www.youtube.com/watch?v=3obEP8eLsCw" },
    { name: "Software Engineering", google: "https://www.geeksforgeeks.org/software-engineering", youtube: "https://www.youtube.com/watch?v=H8erhYQhGz4" },
    { name: "Algorithms", google: "https://www.geeksforgeeks.org/fundamentals-of-algorithms", youtube: "https://www.youtube.com/watch?v=8hly31xKli0" }
  ],

  "IT": [
    { name: "Java", google: "https://www.w3schools.com/java/", youtube: "https://www.youtube.com/watch?v=grEKMHGYyns" },
    { name: "Python", google: "https://www.python.org/about/gettingstarted/", youtube: "https://www.youtube.com/watch?v=_uQrJ0TkZlc" },
    { name: "Object Oriented Programming", google: "https://www.tpointtech.com/java-oops-concepts", youtube: "https://youtu.be/bSrm9RXwBaI?si=RO_m1Y9i44sLZV7W" },
    { name: "Machine Learning", google: "https://www.geeksforgeeks.org/machine-learning", youtube: "https://www.youtube.com/watch?v=GwIo3gDZCVQ" },
    { name: "JavaScript", google: "https://www.w3schools.com/js/", youtube: "https://youtu.be/lfmg-EJ8gm4?si=njXTAXox08nYkPPN" },
    { name: "SQL", google: "https://www.w3schools.com/sql/", youtube: "https://youtu.be/hlGoQC332VM?si=TWYwwsGjxELSU4xM" },
    { name: "C Programming", google: "https://www.learn-c.org/", youtube: "https://www.youtube.com/watch?v=KJgsSFOSQv0" },
    { name: "Blockchain", google: "https://www.geeksforgeeks.org/blockchain", youtube: "https://www.youtube.com/watch?v=SSo_EIwHSd4" },
    { name: "Database Management Systems", google: "https://www.geeksforgeeks.org/database-management-system-dms", youtube: "https://www.youtube.com/watch?v=ztHopE5Wnpc" },
    { name: "Networks", google: "https://www.geeksforgeeks.org/computer-network-tutorials", youtube: "https://www.youtube.com/watch?v=qiQR5rTSshw" }
  ],

  "ME": [
    { name: "Thermodynamics", google: "https://www.engineeringtoolbox.com/thermodynamics-d_101.html", youtube: "https://www.youtube.com/watch?v=ERH8cq4vKXo" },
    { name: "Mechanics", google: "https://www.britannica.com/science/mechanics-physics", youtube: "https://www.youtube.com/watch?v=kKKM8Y-u7ds" },
    { name: "Fluid Mechanics", google: "https://www.engineeringtoolbox.com/fluids-d_150.html", youtube: "https://www.youtube.com/watch?v=Z1WfGpYdb1k" },
    { name: "Heat Transfer", google: "https://www.engineeringtoolbox.com/heat-transfer-d_223.html", youtube: "https://www.youtube.com/watch?v=OA_H9fITjOA" },
    { name: "Strength of Materials", google: "https://www.engineeringtoolbox.com/stress-strain-d_951.html", youtube: "https://www.youtube.com/watch?v=5l-jtqxX6zk" },
    { name: "Manufacturing Processes", google: "https://www.geeksforgeeks.org/manufacturing-processes", youtube: "https://www.youtube.com/watch?v=GQb2wOx93Kw" },
    { name: "Machine Design", google: "https://www.geeksforgeeks.org/machine-design", youtube: "https://www.youtube.com/watch?v=gVXyDPTTRDQ" },
    { name: "Automobile Engineering", google: "https://www.geeksforgeeks.org/automobile-engineering", youtube: "https://www.youtube.com/watch?v=eV2n6JrZzFw" },
    { name: "Vibration Analysis", google: "https://www.engineeringtoolbox.com/vibration-d_905.html", youtube: "https://www.youtube.com/watch?v=cNJt_1vcF3k" },
    { name: "Control Systems", google: "https://www.geeksforgeeks.org/control-systems", youtube: "https://www.youtube.com/watch?v=R0DE_7hBf8Y" }
  ],

  "ECE": [
    { name: "Signals and Systems", google: "https://www.geeksforgeeks.org/signals-and-systems", youtube: "https://www.youtube.com/watch?v=DPLk7eEje5Y" },
    { name: "Electronics", google: "https://www.electronics-tutorials.ws/", youtube: "https://www.youtube.com/watch?v=1I5ZMmrOfnA" },
    { name: "Digital Electronics", google: "https://www.electronics-tutorials.ws/basics/digital-logic-circuit.html", youtube: "https://www.youtube.com/watch?v=4j2X3w6eAow" },
    { name: "Microprocessors", google: "https://www.geeksforgeeks.org/microprocessor", youtube: "https://www.youtube.com/watch?v=J7m1-cGya1g" },
    { name: "Electromagnetic Fields", google: "https://www.geeksforgeeks.org/electromagnetic-field-theory", youtube: "https://www.youtube.com/watch?v=3mhnH7RRGD0" },
    { name: "Communication Systems", google: "https://www.geeksforgeeks.org/communication-systems", youtube: "https://www.youtube.com/watch?v=9ebMwz5ITsM" },
    { name: "Control Systems", google: "https://www.geeksforgeeks.org/control-systems", youtube: "https://www.youtube.com/watch?v=R0DE_7hBf8Y" },
    { name: "Instrumentation", google: "https://www.geeksforgeeks.org/instrumentation", youtube: "https://www.youtube.com/watch?v=m7m_DvVdHhM" },
    { name: "Analog Electronics", google: "https://www.electronics-tutorials.ws/analog/analog-electronics-tutorial.html", youtube: "https://www.youtube.com/watch?v=1I5ZMmrOfnA" },
    { name: "VLSI Design", google: "https://www.geeksforgeeks.org/vlsi-design", youtube: "https://www.youtube.com/watch?v=1sG2DJ0LVzk" }
  ]
};


// Resources route
router.get('/resources', ensureAuthenticated, async (req, res) => {
  const streamSelection = await StreamSelection.findOne({ user_id: req.user.id });

  if (!streamSelection) {
    return res.redirect('/profile');
  }

  const stream = streamSelection.stream_choice;
  const resources = streamResources[stream] || [];

  res.render('resources', { stream, resources });
});

module.exports = router;