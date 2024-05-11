// App.js
import {React, useState, useEffect} from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';

import './App.css'
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';
import Home from './components/Home';
import Profile from './components/Profile';
import Latest from './components/Latest';
import ExperienceDetail from './components/ExperienceDetail';
import Shareexperience from './components/Shareexperience';
import Searchexperience from './components/Searchexperience';
import Aboutus from './components/Aboutus';
import Contactus from './components/Contactus';
import SignIn from './components/SignIn';
import Page2 from './components/Page2';
import UpdateExperience from './components/UpdateExperience';
import EditProfile from './components/EditProfile';
import ChangePassword from './components/ChangePassword';
import CompanyQuestions from './components/CompanyQuestions';
const pageVariants = {
  initial: {
    opacity: 0,
    y: '-100vh' // Start from top of the page
  },
  animate: {
    opacity: 1,
    y: 0, // Slide into view from top
    transition: {
      duration: 0.5,
      ease: 'easeInOut' // Smooth animation
    }
  },
  exit: {
    opacity: 0,
    y: '100vh', // Slide out of view to the bottom
    transition: {
      duration: 0.5,
      ease: 'easeInOut' // Smooth animation
    }
  }
};
const App = () => {
  const [currrollno , setcurrrollno]=useState(null);
  useEffect (()=>{
    console.log("currrollno in app : ", currrollno)
  },[currrollno])
  return (
   
    <Router>
       
        
       {/* <Header /> */}
       
       <div className='body'>
       <Sidebar />
        <div  className='body-style'>
       
      <Routes>
      <Route
    path="/"
    element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}
      >
        <SignIn setcurrrollno={setcurrrollno} />
      </motion.div>
    }
  />
      <Route path="/home" element={
      <motion.div 
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}
     
      ><Home currrollno={currrollno}/></motion.div> } />
        <Route path="/profile" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}   // Adjust duration as needed
      ><Profile currrollno= {currrollno}/></motion.div>} ></Route>
        <Route path="/latest" element={
      <motion.div
     // Adjust duration as needed
      ><Latest /></motion.div>} />
        <Route path="/getexperience/:id" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><ExperienceDetail  currrollno={currrollno}/></motion.div>} />
        <Route path="/updateexperience/:id" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}// Adjust duration as needed
      ><UpdateExperience currrollno={currrollno} /></motion.div>} />
        <Route path="/editprofile/:id" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><EditProfile currrollno={currrollno}/></motion.div>} />
        <Route path="/changepassword/:currrollno" element={
      <motion.div 
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><ChangePassword currrollno={currrollno}/></motion.div>} />
        <Route path="/share-experience" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit" style={{display:"flex", justifyContent:"center"}}
      variants={pageVariants} // Adjust duration as needed
      ><Shareexperience currrollno= {currrollno} /></motion.div>} />
        <Route path="/search-experience" element={
      <motion.div 
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><Searchexperience currrollno={currrollno}/></motion.div>} />
        <Route path="/companyquestions" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><CompanyQuestions currrollno= {currrollno} /></motion.div>} />
        <Route path="/page2" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><Page2 currrollno= {currrollno} /></motion.div>} />
        <Route path="/about-us" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants} // Adjust duration as needed
      ><Aboutus currrollno={currrollno}/></motion.div>} />
        <Route path="/contact-us" element={
      <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}
         // Adjust duration as needed
      ><Contactus currrollno={currrollno}/></motion.div>} />
      </Routes>
      
      </div>
      </div>
      {/* <Footer /> */}
       
    </Router>
 
  );
};

export default App;
