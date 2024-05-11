import React, { useState, useMemo, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Card, Form, Button } from 'react-bootstrap';
import { Doughnut } from 'react-chartjs-2';
import axios from 'axios';
import Sentiment from 'sentiment';

import './Home.css';

const sentiment = new Sentiment();

const calculateSentiment = (experiences) => {
  let totalScore = 0;
  let positiveCount = 0;
  let negativeCount = 0;
  let neutralCount = 0;

  experiences.forEach((experience) => {
    const result = sentiment.analyze(experience.overall_experience);
    totalScore += result.score;

    if (result.score > 0) {
      positiveCount++;
    } else if (result.score < 0) {
      negativeCount++;
    } else {
      neutralCount++;
    }
  });

  const totalCount = experiences.length;
  const positivePercentage = (positiveCount / totalCount) * 100;
  const negativePercentage = (negativeCount / totalCount) * 100;
  const neutralPercentage = (neutralCount / totalCount) * 100;

  return {
    sentimentScore: totalScore / totalCount,
    positivePercentage,
    negativePercentage,
    neutralPercentage,
  };
};

const calculateHiringPercentage = (experiences) => {
  let hiredCount = 0;

  experiences.forEach((experience) => {
    if (experience.placed) {
      hiredCount++;
    }
  });

  const totalCount = experiences.length;
  return (hiredCount / totalCount) * 100;
};

const ExperienceCard = ({ experience }) => {console.log(experience)
  const { company_name, rollno, role_offered, overall_experience, batch , educational_criteria } = experience;
  const truncatedOverallExperience = overall_experience.slice(0, 150) + '...';
  const sentimentResult = sentiment.analyze(overall_experience);
  const sentimentColor = sentimentResult.score > 0 ? 'green' : sentimentResult.score < 0 ? 'red' : 'grey';


  return (
    <div class="card">
   {/*} <Card className="experiencecard" style={{ width: "95%" }}>
      <Card.Body>
        <Card.Title style={{ fontSize: "large", fontWeight: "bold" }}>{company_name}</Card.Title>
        <Card.Text>Batch : {experience.batch}</Card.Text>
        <Card.Text>Role Offered : {role_offered}</Card.Text>
        <Card.Text>{truncatedOverallExperience}</Card.Text>
        <div style={{ color: sentimentColor , paddingBottom:"5px"}}>
          Sentiment: {sentimentResult.score > 0 ? 'Positive' : sentimentResult.score < 0 ? 'Negative' : 'Neutral'}
        </div>
        <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
      </Card.Body>
    </Card>
  */}
  <div class="face face1">
      <div class="content">
        <span class="stars"></span>
        <h2 class="java">{company_name}</h2>
        <p class="java">Educational Criteria : {educational_criteria}</p>
        <p class="java">Overall Experience : {truncatedOverallExperience}</p>
       <Link to={`/getexperience/${experience.experience_id}`}>
          <Button variant="primary">Read More</Button>
        </Link>
      </div>
    </div>
    <div class="face face2">
      <h2>{company_name}</h2>
      <div className='face2-contents'>
      <h4>Role Offered : {role_offered}</h4>
      <h4>Batch : {batch}</h4>
       <div style={{ color: sentimentColor , paddingBottom:"5px"}} className='sentiment'>
          Experience was : {sentimentResult.score > 0 ? 'Positive' : sentimentResult.score < 0 ? 'Negative' : 'Neutral'}
        </div>
        </div>
    </div>
  </div>

  );
};
const Slide = ({ quote }) => {
  return (
    <div className="slide">
      <div className="slide-content">
        <div className="caption">
          <div className="title">Slide title</div>
          <div className="text">
            <p>{quote.text}</p>
          </div>
          <a href="#" className="btn">
            <span className="btn-inner">Learn More</span>
          </a>
        </div>
      </div>
      <div className="image-container">
        <img src={quote.imageUrl} alt="" className="image" />
      </div>
    </div>
  );
};


const QuoteSlider = () => {
 
    const [currentSlide, setCurrentSlide] = useState(0);
    const slides = [
      {
        image: 'https://www.alixbdanthenay.fr/wp-content/uploads/2015/07/Indispensable-1.jpg',
        title: 'Slide title 1',
        description: 'Slide description 1'
      },
      {
        image: 'https://www.alixbdanthenay.fr/wp-content/uploads/2015/07/Indispensable-4-1.jpg',
        title: 'Slide title 2',
        description: 'Slide description 2'
      },
      {
        image: 'https://www.alixbdanthenay.fr/wp-content/uploads/2016/11/11.jpg',
        title: 'Slide title 3',
        description: 'Slide description 3'
      },
      {
        image: 'https://www.alixbdanthenay.fr/wp-content/uploads/2016/11/20mars17-sans-typo.jpg',
        title: 'Slide title 4',
        description: 'Slide description 4'
      }
    ];
  
    useEffect(() => {
      const timeout = setTimeout(() => {
        setCurrentSlide((currentSlide + 1) % slides.length);
      }, 4000);
  
      return () => clearTimeout(timeout);
    }, [currentSlide]);
  
    return (
      <div className="slideshow">
        <div className="slideshow-inner">
          <div className="slides">
            {slides.map((slide, index) => (
              <div key={index} className={`slide ${index === currentSlide ? 'is-active' : ''}`}>
                <div className="slide-content">
                  <div className="caption">
                    <div className="title">{slide.title}</div>
                    <div className="text">
                      <p>{slide.description}</p>
                    </div>
                    <a href="#" className="btn">
                      <span className="btn-inner">Learn More</span>
                    </a>
                  </div>
                </div>
                <div className="image-container">
                  <img src={slide.image} alt="" className="image" />
                </div>
              </div>
            ))}
          </div>
          <div className="pagination">
            {slides.map((slide, index) => (
              <div key={index} className={`item ${index === currentSlide ? 'is-active' : ''}`}>
                <span className="icon">{index + 1}</span>
              </div>
            ))}
          </div>
          <div className="arrows">
          <div className="arrow prev">
  <span className="svg svg-arrow-left">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 26">
      <path d="M13,26c-0.256,0-0.512-0.098-0.707-0.293l-12-12c-0.391-0.391-0.391-1.023,0-1.414l12-12c0.391-0.391,1.023-0.391,1.414,0s0.391,1.023,0,1.414L2.414,13l11.293,11.293c0.391,0.391,0.391,1.023,0,1.414C13.512,25.902,13.256,26,13,26z"/>
    </svg>
    <span className="alt sr-only"></span>
  </span>
</div>
<div className="arrow next">
  <span className="svg svg-arrow-right">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 26">
      <path d="M1,0c0.256,0,0.512,0.098,0.707,0.293l12,12c0.391,0.391,0.391,1.023,0,1.414l-12,12c-0.391,0.391-1.023,0.391-1.414,0s-0.391-1.023,0-1.414L11.586,13L0.293,1.707c-0.391-0.391-0.391-1.023,0-1.414C0.488,0.098,0.744,0,1,0z"/>
    </svg>
    <span className="alt sr-only"></span>
  </span>
</div>
          </div>
        </div>
      </div>
  
  );
};

const Home = ( props) => {
  
  const history = useNavigate();
 
  const [recentExperiences, setRecentExperiences] = useState([]);
  const token = localStorage.getItem("token");
  const {currrollno}=props;
  console.log("currol in home : ", currrollno)
  useEffect(() => {
    if (!currrollno) {
      // Redirect to sign-in if currrollno is null
      history('/');
    }
  }, [currrollno]);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`http://localhost:3001/experiences/experience`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });

        setRecentExperiences(response.data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)));
      } catch (error) {
        console.error("Error fetching experiences:", error);
      }
    };

    fetchData();
  }, []);

  const { sentimentScore, positivePercentage, negativePercentage, neutralPercentage } = useMemo(() => {
    return calculateSentiment(recentExperiences);
  }, [recentExperiences]);

  const hiringPercentage = useMemo(() => {
    return calculateHiringPercentage(recentExperiences);
  }, [recentExperiences]);

  const doughnutData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [{
      data: [positivePercentage, neutralPercentage, negativePercentage],
      backgroundColor: ['green', 'grey', 'red'],
      hoverOffset: 4
    }]
  };

  return (
    <div className="home-container">
      <h1 className="welcome-message">Placement Experience Sharing Portal</h1>
      <div className="quotes-container">
       <QuoteSlider/>
      </div>
      <div className="content-container">
        
        <div>
        <div class="ag-format-container">
  <div class="ag-courses_box">
  <div class="ag-courses_item">
      <a href="#" class="ag-courses-item_link">
        <div class="ag-courses-item_bg"></div>

        <div class="ag-courses-item_title">
        Share your valuable experience such that it is helpful for others.
        </div>

        <div class="ag-courses-item_date-box">
        <Link to="/share-experience" style={{ textDecoration: 'none',  color: 'white' }}>
        <div><h3>Share Experience</h3></div>
      </Link>
        </div>
      </a>
    </div>
    <div class="ag-courses_item">
      <a href="#" class="ag-courses-item_link">
        <div class="ag-courses-item_bg"></div>

        <div class="ag-courses-item_title">
        search for an experience and make your interview journey a success
        </div>

        <div class="ag-courses-item_date-box">
       
      <Link to="/search-experience" style={{ textDecoration: 'none',  color: 'white' }}>
        <div><h3>Search Experience</h3></div>
      </Link>
        </div>
      </a>
    </div>


    
  </div>
</div>
          {/*
        <div className="buttons-container" style={{display:"flex", flexDirection:"row"}}>
          <div style={{borderBlock:"black", backgroundColor:"white", padding:"10px", margin:"10px"}}>
          <h4>Share your valuable experience such that it is helpful for others.</h4>
          <p>Help others share your experience</p>
          <button className="" style={{color : "black", background:"white",borderBlockColor:"black" }}onClick={() => history('/share-experience')}>
            Share Your Experience
          </button>
          </div>
          <div>
            <p>search for an experience and make your interview journey a success</p>
            <p>Gain helpful insights, start by browsing for an experience</p>
          <button className="search-button" onClick={() => history('/search-experience')}>
            Search for an experience
          </button>
          </div>
        </div>*/}
        </div>
        <div className="charts-container">
          <div className="sentiment-chart">
            <Doughnut data={doughnutData} />
            <p className="sentiment-summary">Average sentiment of recent experiences is {sentimentScore.toFixed(2)}</p>
          </div>
          <div className="hiring-chart">
            <Doughnut
              data={{
                labels: ['Hired', 'Not Hired'],
                datasets: [{
                  data: [hiringPercentage, 100 - hiringPercentage],
                  backgroundColor: ['green', 'grey'],
                  hoverOffset: 4
                }]
              }}
            />
            <p className="hiring-summary">Hiring percentage of recent experiences is {hiringPercentage.toFixed(2)}%</p>
          </div>
        </div>
        <div className="recent-experiences-title"><h2>Recent Experiences</h2></div>
        <div className="recent-experiences-container" style={{width : "100%"}}>
          <div className="experience-row">
            {recentExperiences.slice(0,5).map((experience, index) => (
              <div className="experience-card" key={experience.experience_id}>
                <ExperienceCard experience={experience} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
