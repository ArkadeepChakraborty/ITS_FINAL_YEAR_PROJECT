# Intelligent Tutorial System (ITS)
This is Intelligent Tutorial System. this is a smart e-tutor who teaches his student based on their performance on their preliminary test and give course suggestions for themselves. After their all preparations, it takes a final exam of a student and gave a certificate based on their result. Breakdown of this project’s processes: 

•	At First we created the frontend that means the website’s pages of this project using EJS (Embedded Javascript) ans Css (Cascading Style Sheets). Here User (Student) Can register themselves in the website and select his stream too. After stream selection they have to give entrance test based on this knowledge. After giving the test website creates assessment result of the student and forward it to the Python server (Django). Here user can also Update and Delete his account too.

•	For the bankend we use Node.js to create robust APIs to handle user requests and interactions. In Database I used MongoDB Cluster for scalable and secure data storage. Mongoose for managing data schemas and interactions with the database. The aim is to provide a seamless platform for users to register as Students and can get help in their path of learning.

•	When the assessment result comes to the Python Backend, it calculates the right question’s percentage of the user based on the different level of questions (Ex: Easy, Medium, Hard) by XGBoost Algorithm. Then the algorithm gave me the best suggestions from the CSV file. Then the suggestions forwarded to the website again.  

•	Then the coming output suggestions was shown to the user in a proper format to the website.


Skills: EJS(Embedded Javascript),Css(Cascading Style Sheets),Node JS,Postman API,Django,ML Algorithms.
