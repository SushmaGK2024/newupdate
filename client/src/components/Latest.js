// components/Latest.js
// components/ExperienceCard.js
import React,{useState, useEffect} from 'react';
import { Card, Button } from 'react-bootstrap';
import axios from 'axios';
import { Link } from 'react-router-dom';

const ExperienceCard = ({ experience }) => {
  const { title, description, readMore } = experience;
  console.log("experience : ", experience);
  const company=experience.company_name;
  const rollNo= experience.rollno;
  const roleOffered=experience.role_offered;
  const overall_experience=experience.overall_experience;
  const truncatedOverallExperience = overall_experience.slice(0, 50) + '...';
  return (
    <Card className="mb-3">
      <Card.Body>
        <Card.Title>{company}</Card.Title>
        <Card.Text>{truncatedOverallExperience}</Card.Text>
        <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
      </Card.Body>
    </Card>
  );
};






const Latest = () => {
  const [experiences, setExperiences] = useState([]);
  const [loading, setLoading] = useState(true);
  const token= localStorage.getItem("token");
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_BACKEND_URL}/experiences/experience`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setExperiences(response.data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching experiences:', error);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }

  if (!Array.isArray(experiences)) {
    console.error('Experiences is not an array:', experiences);
    return <p>Error fetching experiences</p>;
  }
  return (
    <div>
      <h2>Latest Experiences and Placement Records</h2>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className="experiences" style={{paddingTop:"100px"}}>
           {experiences
          .slice()
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
          .map((experience) => (
            <ExperienceCard key={experience.experience_id} experience={experience} />
          ))}
        </div>
      )}
    </div>
  );
};

export default Latest;