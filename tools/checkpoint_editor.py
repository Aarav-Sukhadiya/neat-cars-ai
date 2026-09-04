import pygame
import json
import sys

def main():
    pygame.init()
    
    try:
        track_img = pygame.image.load("assets/track.png")
    except Exception as e:
        print(f"Error loading track image: {e}")
        sys.exit(1)
        
    width, height = track_img.get_size()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Checkpoint Editor - Left Click: Draw Line | Right Click: Undo | ENTER: Save")
    
    # Checkpoints format: [x1, y1, x2, y2]
    checkpoints = []
    
    # Load existing if available
    try:
        with open('data/checkpoints.json', 'r') as f:
            checkpoints = json.load(f)
    except FileNotFoundError:
        pass

    current_point = None
    running = True

    while running:
        screen.blit(track_img, (0, 0))
        
        # Draw existing checkpoints
        for i, cp in enumerate(checkpoints):
            pygame.draw.line(screen, (0, 255, 0), (cp[0], cp[1]), (cp[2], cp[3]), 3)
            # Draw checkpoint number
            font = pygame.font.SysFont(None, 24)
            text = font.render(str(i + 1), True, (255, 0, 0))
            screen.blit(text, (cp[0], cp[1]))
            
        # Draw line currently being created
        if current_point is not None:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(screen, (0, 255, 255), current_point, mouse_pos, 2)
            
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    if current_point is None:
                        current_point = event.pos
                    else:
                        checkpoints.append([current_point[0], current_point[1], event.pos[0], event.pos[1]])
                        current_point = None
                elif event.button == 3: # Right click (Undo)
                    if current_point is not None:
                        current_point = None
                    elif len(checkpoints) > 0:
                        checkpoints.pop()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    with open('data/checkpoints.json', 'w') as f:
                        json.dump(checkpoints, f, indent=4)
                    print(f"Saved {len(checkpoints)} checkpoints to checkpoints.json!")
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

    pygame.quit()

if __name__ == "__main__":
    main()
