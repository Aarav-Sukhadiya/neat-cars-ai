import pygame
import sys
import json
import math
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/manual_checkpoint_tool.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    pygame.init()

    try:
        track_img = pygame.image.load(img_path).convert_alpha()
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)

    W, H = track_img.get_size()
    info = pygame.display.Info()
    SCREEN_W = min(1280, int(info.current_w * 0.95))
    SCREEN_H = min(720, int(info.current_h * 0.95))

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Manual Checkpoint Editor")
    font = pygame.font.SysFont("monospace", 16)

    cam_x, cam_y = 0, 0
    cam_speed = 10

    checkpoints = []

    # Checkpoint properties
    cp_length = 200
    cp_angle = 0  # degrees

    clock = pygame.time.Clock()
    running = True

    while running:
        keys = pygame.key.get_pressed()
        
        # Camera WASD
        if keys[pygame.K_w]: cam_y -= cam_speed
        if keys[pygame.K_s]: cam_y += cam_speed
        if keys[pygame.K_a]: cam_x -= cam_speed
        if keys[pygame.K_d]: cam_x += cam_speed
        
        # + / - for length
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            cp_length += 3
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            cp_length = max(10, cp_length - 3)
            
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[2]: # Right click hold
            cp_angle += 3
            if cp_angle >= 360:
                cp_angle -= 360

        # Ensure camera stays roughly in bounds
        cam_x = max(-SCREEN_W//2, min(cam_x, W - SCREEN_W//2))
        cam_y = max(-SCREEN_H//2, min(cam_y, H - SCREEN_H//2))
        
        screen.fill((20, 20, 20))
        screen.blit(track_img, (-cam_x, -cam_y))
        
        # Draw existing
        for i, cp in enumerate(checkpoints):
            p1 = (cp[0] - cam_x, cp[1] - cam_y)
            p2 = (cp[2] - cam_x, cp[3] - cam_y)
            # Make the first checkpoint look distinct (starting line)
            color = (0, 255, 0) if i == 0 else (255, 165, 0)
            pygame.draw.line(screen, color, p1, p2, 4)
            
            txt = font.render(str(i), True, (255, 255, 255), (0,0,0))
            cx, cy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            screen.blit(txt, (cx, cy))
            
        # Hover logic
        mx, my = pygame.mouse.get_pos()
        world_mx, world_my = mx + cam_x, my + cam_y
        
        # Calculate hovered line endpoints
        dx = math.cos(math.radians(cp_angle)) * (cp_length / 2)
        dy = math.sin(math.radians(cp_angle)) * (cp_length / 2)
        
        hp1 = (world_mx - dx, world_my - dy)
        hp2 = (world_mx + dx, world_my + dy)
        
        # Draw hover (shifted to screen space)
        sp1 = (hp1[0] - cam_x, hp1[1] - cam_y)
        sp2 = (hp2[0] - cam_x, hp2[1] - cam_y)
        pygame.draw.line(screen, (0, 255, 255), sp1, sp2, 3)
        pygame.draw.circle(screen, (255, 0, 0), (int(sp1[0]), int(sp1[1])), 4) # Mark one end to show direction
        
        # Draw HUD
        hud = [
            "WASD: Move Camera",
            "+/-: Adjust Length",
            "Hold Right Click: Rotate Checkpoint",
            "Left Click: Place Checkpoint",
            "U: Undo last checkpoint",
            "ENTER / S: Save & Exit"
        ]
        
        # Dark HUD background
        pygame.draw.rect(screen, (0,0,0,180), (5, 5, 320, len(hud)*22 + 10))
        for i, text in enumerate(hud):
            rend = font.render(text, True, (255,255,255))
            screen.blit(rend, (10, 10 + i * 22))
            
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left
                    checkpoints.append([int(hp1[0]), int(hp1[1]), int(hp2[0]), int(hp2[1])])
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    if checkpoints:
                        checkpoints.pop()
                elif event.key == pygame.K_RETURN or event.key == pygame.K_s:
                    if not checkpoints:
                        print("No checkpoints added. Exiting without saving.")
                        running = False
                        continue
                        
                    # Figure out a save path based on image name
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = f"data/tracks/{base_name}.json"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    with open(save_path, "w") as f:
                        json.dump(checkpoints, f, indent=4)
                    print(f"Saved {len(checkpoints)} checkpoints to {save_path}!")
                    
                    # Inject into track_config.json
                    config_path = "config/track_config.json"
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                        
                        if "tracks" not in cfg:
                            cfg["tracks"] = {}
                        
                        # Calc start angle based on first two checkpoints
                        if len(checkpoints) > 1:
                            cx0 = (checkpoints[0][0] + checkpoints[0][2]) / 2
                            cy0 = (checkpoints[0][1] + checkpoints[0][3]) / 2
                            cx1 = (checkpoints[1][0] + checkpoints[1][2]) / 2
                            cy1 = (checkpoints[1][1] + checkpoints[1][3]) / 2
                            angle = int(math.degrees(math.atan2(-(cy1 - cy0), cx1 - cx0)))
                        else:
                            angle = 0
                            
                        cfg["tracks"][base_name] = {
                            "name": base_name.replace("_", " ").title(),
                            "image": img_path,
                            "checkpoints": save_path,
                            "max_frames": 7200,
                            "start_pos": [int(checkpoints[0][0]+checkpoints[0][2])//2, int(checkpoints[0][1]+checkpoints[0][3])//2],
                            "start_angle": angle
                        }
                        
                        with open(config_path, "w") as f:
                            json.dump(cfg, f, indent=4)
                        print(f"Updated config/track_config.json with track: {base_name}")
                        
                    running = False
                    sys.exit(0)
                    
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
