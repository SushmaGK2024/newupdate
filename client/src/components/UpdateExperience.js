import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import './UpdateExperience.css';  // Importing CSS file
import { useNavigate } from 'react-router-dom';
const UpdateExperience = (props) => {
  const {currrollno}=props;
  const history = useNavigate();
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno, history]);
  const [experience, setExperience] = useState({});
  const [educationalCriteria, setEducationalCriteria] = useState('');
  const [overallExperience, setOverallExperience] = useState('');
  const [tips, setTips] = useState('');
  const [company_name, setCompanyName] = useState('');
  const [batch, setBatch] = useState('');
  const [role, setRole] = useState('');
  const { id } = useParams();
  const token = localStorage.getItem('token');

  useEffect(() => {
    const fetchExperience = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/getexperience/${id}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setExperience(response.data[0]);
        setEducationalCriteria(response.data[0].educational_criteria);
        setOverallExperience(response.data[0].overall_experience);
        setTips(response.data[0].tips);
        setCompanyName(response.data[0].company_name)
        setBatch(response.data[0].batch)
        setRole(response.data[0].role_offered)
      } catch (error) {
        console.error('Error fetching experience:', error);
      }
    };

    fetchExperience();
  }, [id, token]);

  const handleSaveChanges = async (e) => {
    e.preventDefault();
    
    const config = {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    };

    const updatedData = {
      educationalcriteria: educationalCriteria,
      overall_experience: overallExperience,
      tips: tips
    };

    try {
      const response = await axios.put(`${process.env.REACT_APP_BACKEND_URL}/experiences/experiences/${id}`, updatedData, config);
      console.log("Response:", response.data);
      alert("Updates are saved successfully!!");
    } catch (error) {
      console.error('Error updating experience:', error);
    }
  };

  return (
    <div className="update-experience-container">
      <h2 className="update-experience-title">Update Experience</h2>
      <form onSubmit={handleSaveChanges} className="update-experience-form">
        <div className="form-group">
          <label>Company Name:</label>
          <span className="form-text" style={{marginLeft:"43px"}} >{company_name}</span>
        </div>
        <div className="form-group">
          <label>Batch:</label>
          <span className="form-text" style={{marginLeft:"128px"}}>{batch}</span>
        </div>
        <div className="form-group">
          <label>Role Offered:</label>
          <span className="form-text" style={{marginLeft:"72px"}}>{role}</span>
        </div>
        <div className="form-group" style={{flexDirection:"row"}}>
          <label>Educational Criteria:</label>
          <input 
            type="text"
            value={educationalCriteria}
            onChange={(e) => setEducationalCriteria(e.target.value)}
            className="form-input" style={{width:"60%", marginLeft:"10px"}}
          />
        </div>
        <div className="form-group" style={{display:"flex", flexDirection:"column"}}>
          <label style={{alignSelf:"flex-start"}}>Overall Experience:</label>
          <textarea 
            value={overallExperience}
            onChange={(e) => setOverallExperience(e.target.value)}
            className="form-textarea"
          ></textarea>
        </div>
        <div className="form-group" style={{display:"flex", flexDirection:"column"}}>
          <label style={{alignSelf:"flex-start"}}>Tips:</label>
          <textarea 
            value={tips}
            onChange={(e) => setTips(e.target.value)}
            className="form-textarea" 
          ></textarea>
        </div>
        <button type="submit" className="form-button" style={{marginLeft:"77%"}}>Save Changes</button>
      </form>
    </div>
  );
};

export default UpdateExperience;
