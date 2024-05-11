// components/ExperienceDetail.js
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { Card, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
const ExperienceDetail = (props) => {
  const {currrollno}=props;
  const history = useNavigate();
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno]);
    console.log("in expdetail");
  const { id } = useParams();
  const [experience, setExperience] = useState(null);
  const [loading, setLoading] = useState(true);
  const token = localStorage.getItem('token');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/getexperience/${id}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setExperience(response.data[0]);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching experience:', error);
      }
    };

    fetchData();
  }, [id]);

  if (loading) {
    return <p>Loading...</p>;
  }

  if (!experience) {
    return <p>Error fetching experience</p>;
  }

  console.log("experience : ", experience);
  const company_name=experience.company_name;
  const rollno= experience.rollno;
  const role_offered=experience.role_offered;
  const overall_experience=experience.overall_experience;
const tips= experience.tips;
  return (
    <div  style={{paddingTop:"30px", display:"flex",flexDirection:"column",justifyContent:"center",alignItems:"center", width:"80%", background:"lightsteelblue"

    }}>
      
      <div  style={{display:"flex",justifyContent:"center",alignItems:"center", width : "100%"}}>
        <div style={{width : "90%", padding:"10px"}}>

          <div style={{display:"flex",justifyContent:"center",alignItems:"center"}}><h1>{company_name}</h1></div>
          
          <h3 style={{margin:"10px", float:"right"}}>Added by: {rollno}</h3>
          <h3 style={{margin:"10px", paddingTop:"50px"}}>Role Offered: {role_offered}</h3>
         
          <h3 style={{margin:"10px"}}>Overall Experience : </h3>
          <p style={{fontSize:"large", margin:"10px"}}>{overall_experience}</p>
          <h3>Tips : </h3>
          <p style={{fontSize:"large", margin:"10px"}}>{tips}</p>
          </div>
      </div>
      <br/>
      <Button variant="secondary" style={{marginBottom:"20px"}} onClick={() => window.history.back()}>Back</Button>
    </div>
  );
};

export default ExperienceDetail;