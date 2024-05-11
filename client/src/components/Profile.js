import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Profile.css'
import { Card, Form, Button } from 'react-bootstrap';
import { useNavigate, Link } from 'react-router-dom';



const Profile = (props) => {
  const {currrollno}=props;
  console.log("currrollno in prof : ", props)
    const [user, setUser] = useState(null);
    const [isexpupdated, setisexpupdated]= useState(false);
    const history = useNavigate();
    const handleChangePassword = () => {
      // Navigate to ChangePassword page
      history('/changepassword${currrollno}');
    };
    useEffect(() => {
      if (!currrollno) {
        // Redirect to sign-in if currrollno is null
        history('/');
      } else {
        // Fetch user profile
        const fetchUserProfile = async () => {
          try {
            const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/users/profile/${currrollno}`);
            setUser(response.data);
          } catch (error) {
            console.error('Error fetching user profile:', error);
          }
        };
  
        fetchUserProfile();
      }
    }, [currrollno, history]);
    const handleDeleteExperience = async (experienceId) => {
      const config = {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      };
  
      try {
        const response = await axios.delete(`${process.env.REACT_APP_BACKEND_URL}/experiences/experiences/${experienceId}`, config);
        console.log("Delete Response:", response.data); // Log response data if needed
        alert("Experience deleted successfully!!");
        setisexpupdated(!isexpupdated);
        // Refresh experiences list after deletion or remove the deleted experience from the state
      } catch (error) {
        console.error('Error deleting experience:', error);
      }
    };
  
    const handleLogout = () => {
      // Clear local storage and redirect to sign-in page
      localStorage.removeItem('token');
      localStorage.removeItem('currrollno');
      history('/');
    };
  
  console.log(props);
  
  const token= localStorage.getItem("token");
  console.log(currrollno);
  //const [user, setUser] = useState(null);
  const [academicDetails, setAcademicDetails] = useState(null);
  const [placementRecords, setPlacementRecords] = useState(null);
  const [experiences, setExperiences] = useState(null);
  console.log("in profile");
  useEffect(() => {
    //console.log("rollno : ",rollno)
    
    const fetchUserProfile = async () => {
      try {
        const rollno = currrollno;
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/users/profile/${rollno}`);
       
        setUser(response.data);
        console.log("in fectch")
        console.log("user : ",user);
       // setAcademicDetails(response.data.academic_info);
       // setPlacementRecords(response.data.placementRecords);
        //setExperiences(response.data.experiences);
      } catch (error) {
        console.error('Error fetching user profile:', error);
      }
    };

    fetchUserProfile();

    const fetchExperiences = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/experience/${currrollno}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
        setExperiences(response.data);
      } catch (error) {
        console.error('Error fetching experiences:', error);
      }
    };

    fetchExperiences();

    const fetchPlacementRecords = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/placements/placementrecord/${currrollno}`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setPlacementRecords(response.data);
        console.log("placement records : ",placementRecords)
      } catch (error) {
        console.error('Error fetching placement records:', error);
      }
    };
    fetchPlacementRecords();
  }, [currrollno, isexpupdated]);
  if (!user) {
    return <div>Loading...</div>;
  }
 
 
  return (
    <div style={{width:"100%"}}>
      <div className="profile-container">
      <div style={{display:"flex", flexDirection: "column"}}>
      
      <h1 className="profile-title">User Profile</h1>
      <div style={{marginBottom:"5px"}}>
      <button onClick={handleLogout}style={{ display : "flex",float:"right"}} >Logout</button>
      <Link to={`/changepassword/${currrollno}`}>
      <button style={{ display : "flex",float:"right", marginRight:"10px", }} >Change Password</button>
      </Link>
      
      <Link to={`/editprofile/${currrollno}`}>
          <Button  style={{ display : "flex",float:"right", marginRight:"10px", }}>Edit</Button>
        </Link>
      </div>
      </div>
     <div className="profile-row">
   

      <h2 className="profile-section-title">User Information:</h2>
      <div className='profile-info'>
        <div className="profile-section-row">
        <div className="profile-pic-col" style={{display:"flex",justifyContent:"space-between"}}>
        <img class="profile_image" src="https://www.pngkey.com/png/detail/52-523516_empty-profile-picture-circle.png" alt="thanujathrgyt" />
      
        </div>
        
        <div className="profile-info-username-col">
          <div>Name: {user.full_name} </div>
          <br/>
          <div className="profile-info-item">RollNo: {currrollno}</div>
         
        </div>

      
      </div>
      <hr style={{border: "1px solid #EAEBEF",margin:"20px 0px"}}></hr>
      <div className="profile-section-academic">
        <h3 className="profile-section-title">Academic Details:</h3>
       <div className="profile-info-item">Degree : {user.degree}</div>
        <div className="profile-info-item">Major : {user.major}</div>
        <div className="profile-info-item">Batch : {user.batch}</div>
        <div className="profile-info-item">CGPA : {user.cgpa}</div>
        <div className="profile-info-item">{user.academic_info}</div>

        </div>
      </div>
      </div>
     {/*} <div className="profile-section">
  <h3 className="profile-section-title">Placement Records:</h3>
  {placementRecords && placementRecords.length > 0 ? (
    <ul className="experience-list">
      {placementRecords.map((record, index) => (
        <li key={index} className="experience-item">
          <h3 className="experience-company">Company : {record.company_name}</h3>
          <div className="profile-info-item">Role : {record.role}</div>
          <div className="profile-info-item">CTC : {record.ctc}</div>
        </li>
      ))}
    </ul>
  ) : (
    <div className="profile-info-item">No placement records found.</div>
  )}
</div>
  */}
    <hr style={{border: "1px solid #EAEBEF",margin:"20px 0px", width:"100%"}}></hr>
      
      <div className="profile-section-experience">
        <h3 className="profile-section-title">Experiences</h3>
        {experiences && experiences.length > 0 ? (
          <ul className="experience-list">
            {experiences.map((experience) => (
               <ExperienceCard 
               key={experience.experience_id} 
               experience={experience} 
               onDelete={() => handleDeleteExperience(experience.experience_id)} // Pass handleDeleteExperience function as prop
             />
               ))}
          </ul>
        ) : (
          <div className="profile-info-item">No experiences found.</div>
        )}
      </div>
    </div>
    </div>
  );
};
const ExperienceCard = ({ experience, onDelete }) => {
  const { title, description, readMore } = experience;
  const company = experience.company_name;
  const rollNo = experience.rollno;
  const roleOffered = experience.role_offered;
  const overall_experience = experience.overall_experience;
  const educational_criteria= experience.educational_criteria;
  const batch = experience.batch;
  const truncatedOverallExperience = overall_experience.slice(0, 150) + '...';

 /* const sentimentResult = sentiment.analyze(experience.overall_experience);
  const sentimentColor = sentimentResult.score > 0 ? 'green' : sentimentResult.score < 0 ? 'red' : 'grey';

  const difficultyLevel = getDifficultyLevel([experience]);
  let difficultyGradient;

  switch (difficultyLevel) {
    case 'Easy':
      difficultyGradient = 'linear-gradient(to right, #b9e68c, #69db7c)';
      break;
    case 'Moderate':
      difficultyGradient = 'linear-gradient(to right, #ffeaa7, #ff9f43)';
      break;
    case 'Difficult':
      difficultyGradient = 'linear-gradient(to right, #ff7979, #eb4d4b)';
      break;
    default:
      difficultyGradient = 'linear-gradient(to right, #dcdde1, #b2bec3)';
      break;
  }
*/
  return (
    <div className='card'>
    {/*<Card className="experiencecard" style={{ width: "95%" }}>
      <Card.Body style={{width : "100%"}}>
        <Card.Title style={{ fontSize: "large", fontWeight: "bold" }}>{company}</Card.Title>
        <Card.Text>Batch : {experience.batch}</Card.Text>
        <Card.Text>Role Offered : {roleOffered}</Card.Text>
        <Card.Text>{truncatedOverallExperience}</Card.Text>
        *<div style={{ color: sentimentColor }}>
          Sentiment: {sentimentResult.score > 0 ? 'Positive' : sentimentResult.score < 0 ? 'Negative' : 'Neutral'}
        </div>
        <div style={{ marginTop: '10px', marginBottom:'8px',display: 'flex', alignItems: 'center' }}>
          Difficulty : {difficultyLevel}
          <span 
            style={{ 
              backgroundImage: difficultyGradient,
              width: '50px', 
              height: '10px', 
              marginLeft: '10px', 
              display: 'inline-block',
              borderRadius: '5px'
            }}>
          </span>
          </div>
          <div>
        <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
        
          <Button  variant="danger" onClick={onDelete} style={{float: "right"}}>Delete</Button>
        
        <Link to={`/updateexperience/${experience.experience_id}`}>
          <Button variant="primary" style={{float: "right"}}>Update</Button>
        </Link>
        </div>
      </Card.Body>
        </Card>*/}
        <div class="face face1">
      <div class="content">
        <span class="stars"></span>
        <h2 class="java">{company}</h2>
        <p class="java">Educational Criteria : {educational_criteria}</p>
        <p class="java">Overall Experience : {truncatedOverallExperience}</p>
       <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
       
        
        <Button  variant="danger" onClick={onDelete} style={{float: "right"}}>Delete</Button>
      
      <Link to={`/updateexperience/${experience.experience_id}`}>
        <Button variant="primary" style={{float: "right"}}>Update</Button>
      </Link>
      </div>
    </div>
    <div class="face face2">
      <h2>{company}</h2>
      <div className='face2-contents'>
      <h4>Role Offered : {roleOffered}</h4>
      <h4>Batch : {batch}</h4>
       
        </div>
    </div>
        </div>
  );
};
export default Profile;