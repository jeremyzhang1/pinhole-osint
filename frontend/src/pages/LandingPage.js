import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Container from 'react-bootstrap/Container'
import Navbar from 'react-bootstrap/Navbar'
import Nav from 'react-bootstrap/Nav'
import Button from 'react-bootstrap/Button'
import Form from 'react-bootstrap/Form'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'

import Footer from '../utils/Footer'

import 'bootstrap/dist/css/bootstrap.min.css';
import '../styles/LandingPage.css';

export const baseUrl = "http://localhost:8000"

function LandingPage() {
    const [originalPrompt, setOriginalPrompt] = useState("")
    const [originalOutput, setOriginalOutput] = useState("")

    const [loadingOne, setLoadingOne] = useState(false)
    const [loadingTwo, setLoadingTwo] = useState(false)

    function handleSubmit(e) {
        e.preventDefault()
        setLoadingOne(true)
        setLoadingTwo(true)
        setOriginalOutput("")

        fetch(baseUrl + '/client/get-original-response?' + new URLSearchParams({
            request: originalPrompt
        }))
            .then(results => results.json())
            .then(data => {
                setOriginalOutput(data)
                setLoadingOne(false)
            })
        fetch(baseUrl + '/client/get-jailbroken-response?' + new URLSearchParams({
            request: originalPrompt
        }))
            .then(results => results.json())
            .then(data => {
                setLoadingTwo(false)
            })
    }

    return (
        <div>
            <div className='home'>
                <Container>
                    <Navbar className='navbar-no-bg'>
                        <Navbar.Brand href="/" className='landing-text'>📸 <b>Pinhole OSINT</b></Navbar.Brand>
                    </Navbar>
                    <div className='landing-text'>
                        <h1 className='h1-name' style={{ marginTop: "10%" }}>
                            <strong>
                                OSINT in <span style={{ color: "orange" }}>three dimensions</span>
                            </strong>
                        </h1>
                        <p className='h2-slogan' style={{ marginTop: "3%" }}>
                            Pinhole employs an agentic approach to providing rich, <br />three dimensional context for the intelligence community.
                        </p>
                        <div style={{ marginTop: '2%' }}>
                            <a href="/#features" className='no-style-link'>
                                <Button variant='light' className="get-started-button" style={{ margin: '8px' }}>
                                    Get Started
                                </Button>
                            </a>
                        </div>
                    </div>
                </Container>
            </div>
            <Container id="features">
                <h1 className="text-center h1-action">Enter an IP address:</h1>
                <Form>
                    <Form.Group className="mb-3" controlId="exampleForm.ControlTextarea1">
                        <Form.Label>IP Address:</Form.Label>
                        <Form.Control as="textarea" rows={3} onChange={(e) => setOriginalPrompt(e.target.value)} />
                    </Form.Group>
                </Form>
                <Button onClick={(e) => handleSubmit(e)} disabled={loadingOne || loadingTwo} variant='danger'>Submit</Button>
                <h1 className='h1-action'>OSINT Search Results:</h1>
                <br />
                <br />
                <Row>
                    <Col md={6}>
                        <h3>Placeholder</h3>
                        <Form.Control as="textarea" rows={5} value={originalPrompt} />
                        <br />
                        <h3>Placeholder</h3>
                        <Form.Control as="textarea" rows={10} value={originalOutput} />
                    </Col>
                    <Col md={6}>
                        <h3>Placeholder</h3>
                    </Col>
                </Row>
            </Container>
            <Footer />
        </div>
    )
}

export default LandingPage
