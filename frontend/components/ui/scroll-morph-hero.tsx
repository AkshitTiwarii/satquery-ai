"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { motion, useTransform, useSpring, useScroll, useMotionValue } from "framer-motion";

export type AnimationPhase = "scatter" | "line" | "circle" | "bottom-strip";

interface FlipCardProps {
    src: string;
    index: number;
    total: number;
    phase: AnimationPhase;
    isMobile: boolean;
    target: { x: number; y: number; rotation: number; scale: number; opacity: number };
}

function FlipCard({
    src,
    index,
    total,
    phase,
    isMobile,
    target,
}: FlipCardProps) {
    const cardWidth = isMobile ? 48 : 64;
    const cardHeight = isMobile ? 68 : 90;

    return (
        <motion.div
            animate={{
                x: target.x,
                y: target.y,
                rotate: target.rotation,
                scale: target.scale,
                opacity: target.opacity,
            }}
            transition={{
                type: "spring",
                stiffness: 45,
                damping: 18,
            }}
            style={{
                position: "absolute",
                width: cardWidth,
                height: cardHeight,
                transformStyle: "preserve-3d",
                perspective: "1000px",
            }}
            className="cursor-pointer group select-none touch-manipulation"
        >
            <motion.div
                className="relative h-full w-full"
                style={{ transformStyle: "preserve-3d" }}
                transition={{ duration: 0.6, type: "spring", stiffness: 260, damping: 20 }}
                whileHover={{ rotateY: 180, scale: 1.1 }}
                whileTap={{ rotateY: 180, scale: 1.05 }}
            >
                {/* Front Face */}
                <div
                    className="absolute inset-0 h-full w-full overflow-hidden rounded-xl sm:rounded-2xl shadow-xl bg-slate-900 border border-white/15"
                    style={{ backfaceVisibility: "hidden" }}
                >
                    <img
                        src={src}
                        alt={`satellite-vis-${index}`}
                        className="h-full w-full object-cover"
                        loading="lazy"
                    />
                    <div className="absolute inset-0 bg-black/10 transition-colors group-hover:bg-transparent" />
                </div>

                {/* Back Face */}
                <div
                    className="absolute inset-0 h-full w-full overflow-hidden rounded-xl sm:rounded-2xl shadow-xl bg-[#05080b] flex flex-col items-center justify-center p-2 sm:p-3 border border-white/20"
                    style={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)" }}
                >
                    <div className="text-center">
                        <p className="text-[7px] sm:text-[8px] font-bold text-[#8eb7ff] uppercase tracking-widest mb-0.5">Sensor</p>
                        <p className="text-[10px] sm:text-xs font-medium text-white">#{index + 1}</p>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
}

const TOTAL_IMAGES = 20;

const IMAGES = [
    "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=300&q=80",
    "https://images.unsplash.com/photo-1519710164239-da123dc03ef4?w=300&q=80",
    "https://images.unsplash.com/photo-1497366216548-37526070297c?w=300&q=80",
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=300&q=80",
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=300&q=80",
    "https://images.unsplash.com/photo-1506765515384-028b60a970df?w=300&q=80",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=300&q=80",
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=300&q=80",
    "https://images.unsplash.com/photo-1500485035595-cbe6f645feb1?w=300&q=80",
    "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=300&q=80",
    "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=300&q=80",
    "https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9?w=300&q=80",
    "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=300&q=80",
    "https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?w=300&q=80",
    "https://images.unsplash.com/photo-1493246507139-91e8fad9978e?w=300&q=80",
    "https://images.unsplash.com/photo-1494438639946-1ebd1d20bf85?w=300&q=80",
    "https://images.unsplash.com/photo-1483729558449-99ef09a8c325?w=300&q=80",
    "https://images.unsplash.com/photo-1518173946687-a4c8892bbd9f?w=300&q=80",
    "https://images.unsplash.com/photo-1523961131990-5ea7c61b2107?w=300&q=80",
    "https://images.unsplash.com/photo-1496568816309-51d7c20e3b21?w=300&q=80",
];

const lerp = (start: number, end: number, t: number) => start * (1 - t) + end * t;

export default function IntroAnimation() {
    const [introPhase, setIntroPhase] = useState<AnimationPhase>("scatter");
    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    const isMobile = containerSize.width > 0 && containerSize.width < 768;

    // Natural page scroll integration
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start end", "end start"]
    });

    useEffect(() => {
        if (!containerRef.current) return;

        const updateSize = () => {
            if (containerRef.current) {
                setContainerSize({
                    width: containerRef.current.offsetWidth,
                    height: containerRef.current.offsetHeight,
                });
            }
        };

        const observer = new ResizeObserver(updateSize);
        observer.observe(containerRef.current);
        updateSize();

        return () => observer.disconnect();
    }, []);

    // Natural Scroll Mapping
    const morphProgress = useTransform(scrollYProgress, [0.12, 0.52], [0, 1]);
    const smoothMorph = useSpring(morphProgress, { stiffness: 45, damping: 20 });

    const scrollRotate = useTransform(scrollYProgress, [0.45, 0.9], [0, 360]);
    const smoothScrollRotate = useSpring(scrollRotate, { stiffness: 45, damping: 20 });

    const mouseX = useMotionValue(0);
    const smoothMouseX = useSpring(mouseX, { stiffness: 30, damping: 20 });

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!containerRef.current) return;
            const rect = containerRef.current.getBoundingClientRect();
            if (e.clientY >= rect.top && e.clientY <= rect.bottom) {
                const relativeX = e.clientX - rect.left;
                const normalizedX = (relativeX / rect.width) * 2 - 1;
                mouseX.set(normalizedX * (isMobile ? 40 : 90));
            }
        };
        window.addEventListener("mousemove", handleMouseMove);
        return () => window.removeEventListener("mousemove", handleMouseMove);
    }, [mouseX, isMobile]);

    useEffect(() => {
        const timer1 = setTimeout(() => setIntroPhase("line"), 300);
        const timer2 = setTimeout(() => setIntroPhase("circle"), 1200);
        return () => {
            clearTimeout(timer1);
            clearTimeout(timer2);
        };
    }, []);

    const scatterPositions = useMemo(() => {
        return IMAGES.map(() => ({
            x: (Math.random() - 0.5) * (isMobile ? 500 : 1200),
            y: (Math.random() - 0.5) * (isMobile ? 350 : 700),
            rotation: (Math.random() - 0.5) * 160,
            scale: 0.6,
            opacity: 0,
        }));
    }, [isMobile]);

    const [morphValue, setMorphValue] = useState(0);
    const [rotateValue, setRotateValue] = useState(0);
    const [parallaxValue, setParallaxValue] = useState(0);

    useEffect(() => {
        const unMorph = smoothMorph.on("change", setMorphValue);
        const unRotate = smoothScrollRotate.on("change", setRotateValue);
        const unParallax = smoothMouseX.on("change", setParallaxValue);
        return () => {
            unMorph();
            unRotate();
            unParallax();
        };
    }, [smoothMorph, smoothScrollRotate, smoothMouseX]);

    const contentOpacity = useTransform(smoothMorph, [0.65, 0.95], [0, 1]);
    const contentY = useTransform(smoothMorph, [0.65, 0.95], [20, 0]);

    return (
        <div ref={containerRef} className="relative w-full h-[520px] sm:h-[600px] md:h-[720px] bg-[#05080b] overflow-hidden flex flex-col items-center justify-center">
            {/* Intro Text (Fades out smoothly on scroll) */}
            <div className="absolute z-0 flex flex-col items-center justify-center text-center pointer-events-none top-1/2 -translate-y-1/2 px-4 w-full max-w-lg">
                <motion.h2
                    initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
                    animate={introPhase === "circle" && morphValue < 0.5 ? { opacity: 1 - morphValue * 2, y: 0, filter: "blur(0px)" } : { opacity: 0, filter: "blur(10px)" }}
                    transition={{ duration: 0.9 }}
                    className="font-display text-xl sm:text-3xl md:text-4xl font-medium tracking-tight text-white"
                >
                    The future is built on AI.
                </motion.h2>
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={introPhase === "circle" && morphValue < 0.5 ? { opacity: 0.6 - morphValue } : { opacity: 0 }}
                    transition={{ duration: 0.9, delay: 0.15 }}
                    className="mt-3 sm:mt-4 font-mono text-[10px] sm:text-xs font-bold tracking-[0.2em] text-[#8eb7ff] uppercase"
                >
                    SCROLL TO EXPLORE
                </motion.p>
            </div>

            {/* Arc Active Content (Fades in smoothly as arc forms) */}
            <motion.div
                style={{ opacity: contentOpacity, y: contentY }}
                className="absolute top-[6%] sm:top-[8%] md:top-[10%] z-10 flex flex-col items-center justify-center text-center pointer-events-none px-5 max-w-xl"
            >
                <h2 className="font-display text-2xl sm:text-4xl md:text-5xl font-semibold text-white tracking-tight mb-2 sm:mb-4">
                    Explore Our Vision
                </h2>
                <p className="text-xs sm:text-sm md:text-base text-white/60 leading-relaxed max-w-md">
                    Discover a world where technology meets creativity. <br className="hidden sm:block" />
                    Scroll through our curated collection of innovations designed to shape the future.
                </p>
            </motion.div>

            {/* 3D Cards Stage */}
            <div className="relative flex items-center justify-center w-full h-full">
                {IMAGES.slice(0, TOTAL_IMAGES).map((src, i) => {
                    let target = { x: 0, y: 0, rotation: 0, scale: 1, opacity: 1 };

                    if (introPhase === "scatter") {
                        target = scatterPositions[i];
                    } else if (introPhase === "line") {
                        const lineSpacing = isMobile ? 42 : 72;
                        const lineTotalWidth = TOTAL_IMAGES * lineSpacing;
                        const lineX = i * lineSpacing - lineTotalWidth / 2;
                        target = { x: lineX, y: 0, rotation: 0, scale: 1, opacity: 1 };
                    } else {
                        const minDimension = Math.min(containerSize.width || 400, containerSize.height || 500);

                        const circleRadius = isMobile 
                            ? Math.min(minDimension * 0.38, 160)
                            : Math.min(minDimension * 0.35, 340);
                            
                        const circleAngle = (i / TOTAL_IMAGES) * 360;
                        const circleRad = (circleAngle * Math.PI) / 180;
                        const circlePos = {
                            x: Math.cos(circleRad) * circleRadius,
                            y: Math.sin(circleRad) * circleRadius,
                            rotation: circleAngle + 90,
                        };

                        const baseRadius = Math.min(containerSize.width || 400, (containerSize.height || 500) * 1.5);
                        const arcRadius = baseRadius * (isMobile ? 1.05 : 1.15);
                        const arcApexY = (containerSize.height || 500) * (isMobile ? 0.38 : 0.26);
                        const arcCenterY = arcApexY + arcRadius;

                        const spreadAngle = isMobile ? 95 : 130;
                        const startAngle = -90 - (spreadAngle / 2);
                        const step = spreadAngle / (TOTAL_IMAGES - 1);

                        const scrollProgress = Math.min(Math.max(rotateValue / 360, 0), 1);
                        const maxRotation = spreadAngle * 0.75;
                        const boundedRotation = -scrollProgress * maxRotation;

                        const currentArcAngle = startAngle + (i * step) + boundedRotation;
                        const arcRad = (currentArcAngle * Math.PI) / 180;

                        const arcPos = {
                            x: Math.cos(arcRad) * arcRadius + parallaxValue,
                            y: Math.sin(arcRad) * arcRadius + arcCenterY,
                            rotation: currentArcAngle + 90,
                            scale: isMobile ? 1.1 : 1.7,
                        };

                        target = {
                            x: lerp(circlePos.x, arcPos.x, morphValue),
                            y: lerp(circlePos.y, arcPos.y, morphValue),
                            rotation: lerp(circlePos.rotation, arcPos.rotation, morphValue),
                            scale: lerp(1, arcPos.scale, morphValue),
                            opacity: 1,
                        };
                    }

                    return (
                        <FlipCard
                            key={i}
                            src={src}
                            index={i}
                            total={TOTAL_IMAGES}
                            phase={introPhase}
                            isMobile={isMobile}
                            target={target}
                        />
                    );
                })}
            </div>
        </div>
    );
}
