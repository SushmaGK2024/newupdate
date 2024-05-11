// components/SignIn.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const SignIn = ({setcurrrollno}) => {
    console.log("succes")
  const [rollNo, setRollNo] = useState('');
  const [password, setPassword] = useState('');
  const history = useNavigate();

  const handleSignIn = async () => {
    // Perform API call to authenticate user
    // Use the entered rollNo and password
console.log("geting")
    try {
      const response = await fetch('http://localhost:3001/users/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          rollNo,
          password,
        }),
      });

      const data = await response.json();
       console.log(data);
      if (response.ok) {
        console.log("scuccess")
        // Authentication successful, redirect to home page or profile
        setcurrrollno(rollNo);
        console.log(rollNo);
        history('/home');
        console.log("in home"); // Replace '/home' with the route you want to redirect to
        const token = data.token;
        const userId =data.userId;
        localStorage.setItem('token', token);
        localStorage.setItem('userId', userId);
        console.log("token signin : ",localStorage.getItem('token'));
      } else {
        // Authentication failed, handle error (display error message, etc.)
        console.error('Authentication failed:', data);
      }
    } catch (error) {
      console.error('Error during sign-in:', error);
    }
  };

  return (
    <div  style={{display:"flex",flexDirection:"column",paddingTop:"50px",paddingRight:"80px", justifyContent:"center", alignItems:"center", height:"65vh", marginBottom:"10%"}}>
      <h2>Sign In</h2>
      <div>
      <label>
        Roll No:
        <input type="text" value={rollNo} onChange={(e) => setRollNo(e.target.value)} />
      </label>
      <br />
      <label>
        Password:
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      </label>
      <br />
      <button onClick={handleSignIn}>Sign In</button>
      </div>
    </div>
  );
};

export default SignIn;
