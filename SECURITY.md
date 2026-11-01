# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0.0 | :x:                |

## Reporting a Vulnerability

We take the security and privacy of **FaceMatch JS** seriously. Because this application processes facial images and biometric vector descriptors directly within the user's browser, client-side isolation and data integrity are top priorities.

### Privacy Guarantee
* All face detection, landmark calculation, and descriptor matching are executed **100% client-side** using WebGL/WASM via TensorFlow.js.
* No images, biometric vectors, or telemetry data are collected or transmitted to any remote server.

### Vulnerability Disclosure
If you discover a security vulnerability, model integrity flaw, or unauthorized data leak:
1. **Do not create a public GitHub issue.**
2. Send an email to the lead maintainer at **krupalmalakiya@gmail.com** with details of the vulnerability.
3. Include:
   * Description of the potential vulnerability.
   * Reproduction steps or proof-of-concept.
   * Expected impact.
4. We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline.
