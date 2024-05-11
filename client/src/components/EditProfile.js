import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import './EditProfile.css';  // Importing CSS file
import { useNavigate } from 'react-router-dom';
const EditProfile = (props) => {
  const {currrollno}=props;
  const history = useNavigate();
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno]);
  const [profile, setProfile] = useState({});
  const [name, setName] = useState('');
  const [degree, setDegree] = useState('');
  const [major, setMajor] = useState('');
  const [batch, setBatch] = useState('');
  const [cgpa, setCgpa] = useState('');
  const { id } = useParams();
  const rollno= id;
  console.log("rollno : ", id)
  const token = localStorage.getItem('token');

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/users/profile/${id}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        console.log("user data : ", response.data)
        setProfile(response.data);
        setName(response.data.full_name);
        setDegree(response.data.degree);
        setMajor(response.data.major);
        setBatch(response.data.batch);
        setCgpa(response.data.cgpa);
      } catch (error) {
        console.error('Error fetching profile:', error);
      }
    };

    fetchProfile();
  }, [id, token]);

  const handleSaveChanges = async (e) => {
    e.preventDefault();
    
    const config = {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    };
 
    const updatedData = {
      full_name: name,
      degree: degree,
      major: major,
      batch: batch,
      cgpa: cgpa
    };
    console.log("updated data : ", updatedData)
    try {
      const response = await axios.put(`${process.env.REACT_APP_BACKEND_URL}/users/profile/${id}`, updatedData, config);
      console.log("Response:", response.data);
      alert("Updates are saved successfully!!");
    } catch (error) {
      console.error('Error updating profile:', error);
    }
  };

  return (
    <div className="edit-profile-container">
      <h2 className="edit-profile-title">Edit Profile</h2>
      <form onSubmit={handleSaveChanges} className="edit-profile-form">
        <div className="form-group">
          <label>Roll No:</label>
          <span className="form-text">{rollno}</span>
        </div>
        <div className="form-group">
          <label>Name:</label>
          <input 
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="form-input"
          />
        </div>
        <div className="form-group">
          <label>Degree:</label>
          <input 
            type="text"
            value={degree}
            onChange={(e) => setDegree(e.target.value)}
            className="form-input-degree"
          />
        </div>
        <div className="form-group">
          <label>Major:</label>
          <input 
            type="text"
            value={major}
            onChange={(e) => setMajor(e.target.value)}
            className="form-input"
          />
        </div>
        <div className="form-group">
          <label>Batch:</label>
          <input 
            type="text"
            value={batch}
            onChange={(e) => setBatch(e.target.value)}
            className="form-input"
          />
        </div>
        <div className="form-group">
          <label>CGPA:</label>
          <input 
            type="text"
            value={cgpa}
            onChange={(e) => setCgpa(e.target.value)}
            className="form-input"
          />
        </div>
        <button type="submit" className="form-button">Save Changes</button>
      </form>
    </div>
  );
};

export default EditProfile;
