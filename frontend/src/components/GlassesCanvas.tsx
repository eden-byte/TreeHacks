"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export default function GlassesCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const glassesGroupRef = useRef<THREE.Group | null>(null);

  const [isRotating, setIsRotating] = useState(true);
  const [rotationSpeed] = useState(1.0);
  const frameColor = "#000000";

  // mouse refs
  const mouseX = useRef(0);
  const mouseY = useRef(0);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;

    // Scene and camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 2, 11);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dir1.position.set(5, 10, 5);
    dir1.castShadow = true;
    scene.add(dir1);

    const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dir2.position.set(-5, 5, -5);
    scene.add(dir2);

    // Material
    const frameMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color(frameColor),
      metalness: 0.3,
      roughness: 0.4,
    });

    // Group
    const glassesGroup = new THREE.Group();
    glassesGroupRef.current = glassesGroup;

    // Helper to create rounded-rectangle frame with hole (extruded)
    function createFrameWithHole(outerWidth: number, outerHeight: number, innerWidth: number, innerHeight: number, depth: number, radius: number) {
      const outerShape = new THREE.Shape();
      const x = -outerWidth / 2;
      const y = -outerHeight / 2;

      outerShape.moveTo(x + radius, y);
      outerShape.lineTo(x + outerWidth - radius, y);
      outerShape.quadraticCurveTo(x + outerWidth, y, x + outerWidth, y + radius);
      outerShape.lineTo(x + outerWidth, y + outerHeight - radius);
      outerShape.quadraticCurveTo(x + outerWidth, y + outerHeight, x + outerWidth - radius, y + outerHeight);
      outerShape.lineTo(x + radius, y + outerHeight);
      outerShape.quadraticCurveTo(x, y + outerHeight, x, y + outerHeight - radius);
      outerShape.lineTo(x, y + radius);
      outerShape.quadraticCurveTo(x, y, x + radius, y);

      const holePath = new THREE.Path();
      const hx = -innerWidth / 2;
      const hy = -innerHeight / 2;
      const holeRadius = radius * 0.85;

      holePath.moveTo(hx + holeRadius, hy);
      holePath.lineTo(hx + innerWidth - holeRadius, hy);
      holePath.quadraticCurveTo(hx + innerWidth, hy, hx + innerWidth, hy + holeRadius);
      holePath.lineTo(hx + innerWidth, hy + innerHeight - holeRadius);
      holePath.quadraticCurveTo(hx + innerWidth, hy + innerHeight, hx + innerWidth - holeRadius, hy + innerHeight);
      holePath.lineTo(hx + holeRadius, hy + innerHeight);
      holePath.quadraticCurveTo(hx, hy + innerHeight, hx, hy + innerHeight - holeRadius);
      holePath.lineTo(hx, hy + holeRadius);
      holePath.quadraticCurveTo(hx, hy, hx + holeRadius, hy);

      outerShape.holes.push(holePath);

      const extrudeSettings: THREE.ExtrudeGeometryOptions = {
        depth,
        bevelEnabled: false,
        bevelThickness: 0,
        bevelSize: 0,
        bevelSegments: 0,
      };

      return new THREE.ExtrudeGeometry(outerShape, extrudeSettings);
    }

    // Left and right lens frames (rounded)
    const leftLensGeometry = createFrameWithHole(4.5, 3.5, 3.8, 2.8, 0.3, 1.2);
    const leftLensMesh = new THREE.Mesh(leftLensGeometry, frameMaterial);
    leftLensMesh.position.set(-3, 0, 0);

    const rightLensGeometry = createFrameWithHole(4.5, 3.5, 3.8, 2.8, 0.3, 1.2);
    const rightLensMesh = new THREE.Mesh(rightLensGeometry, frameMaterial);
    rightLensMesh.position.set(3, 0, 0);

    // Bridge with nose arch
    function createBridgeWithNoseArch() {
      const bridgeGroup = new THREE.Group();
      const bridgeShape = new THREE.Shape();
      bridgeShape.moveTo(-0.9, 1.4);
      bridgeShape.lineTo(0.9, 1.4);
      bridgeShape.lineTo(0.9, 0.5);
      bridgeShape.quadraticCurveTo(0.5, 0.35, 0, 0.3);
      bridgeShape.quadraticCurveTo(-0.5, 0.35, -0.9, 0.5);
      bridgeShape.lineTo(-0.9, 1.4);

      const extrudeSettings: THREE.ExtrudeGeometryOptions = {
        depth: 0.3,
        bevelEnabled: true,
        bevelThickness: 0.08,
        bevelSize: 0.05,
        bevelSegments: 3,
      };

      const bridgeGeometry = new THREE.ExtrudeGeometry(bridgeShape, extrudeSettings);
      const bridgeMesh = new THREE.Mesh(bridgeGeometry, frameMaterial);
      bridgeGroup.add(bridgeMesh);
      return bridgeGroup;
    }

    const bridgeMesh = createBridgeWithNoseArch();

    // Temples
    const templeGeometry = new THREE.BoxGeometry(6, 0.6, 0.25);
    const leftTempleMesh = new THREE.Mesh(templeGeometry, frameMaterial);
    leftTempleMesh.position.set(-5.5, 0, -3);
    leftTempleMesh.rotation.y = Math.PI / 2;

    const rightTempleMesh = new THREE.Mesh(templeGeometry, frameMaterial);
    rightTempleMesh.position.set(5.5, 0, -3);
    rightTempleMesh.rotation.y = Math.PI / 2;

    // Hinges
    const hingeGeometry = new THREE.SphereGeometry(0.35, 16, 16);
    const leftHingeMesh = new THREE.Mesh(hingeGeometry, frameMaterial);
    leftHingeMesh.position.set(-5.5, 0, 0);
    const rightHingeMesh = new THREE.Mesh(hingeGeometry, frameMaterial);
    rightHingeMesh.position.set(5.5, 0, 0);

    // Add parts
    glassesGroup.add(leftLensMesh, rightLensMesh, bridgeMesh, leftTempleMesh, rightTempleMesh, leftHingeMesh, rightHingeMesh);
    scene.add(glassesGroup);



    // Mouse interaction
    const onMouseMove = (event: MouseEvent) => {
      mouseX.current = (event.clientX / window.innerWidth) * 2 - 1;
      mouseY.current = -(event.clientY / window.innerHeight) * 2 + 1;
    };

    window.addEventListener("mousemove", onMouseMove);

    // Resize
    const onResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", onResize);

    // Animation loop
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);

      if (isRotating) {
        glassesGroup.rotation.y += 0.01 * rotationSpeed;
      }

      const targetRotationX = mouseY.current * 0.3;
      const targetRotationY = mouseX.current * 0.5;

      // Smooth follow
      glassesGroup.rotation.x += (targetRotationX - glassesGroup.rotation.x) * 0.05;
      if (!isRotating) {
        glassesGroup.rotation.y += (targetRotationY - glassesGroup.rotation.y) * 0.05;
      }

      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", onResize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      renderer.dispose();
      if (container && renderer.domElement.parentElement === container) container.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [/* initial mount only */]);



  return (
    <div style={{ position: "relative", width: "100%", height: "480px" }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
