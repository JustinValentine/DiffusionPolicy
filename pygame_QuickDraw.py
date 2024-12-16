import pygame
import time
import csv

def record_doodle(doodle_class):
    pygame.init()

    WIDTH, HEIGHT = 256, 256
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Doodle Recorder: {doodle_class}")

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    doodle_data = []

    running = True

    last_capture_time = time.time()
    capture_interval = 0.1  # seconds

    is_drawing = False  

    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get mouse state
        x, y = pygame.mouse.get_pos()
        x = max(0, min(WIDTH - 1, x))  
        y = max(0, min(HEIGHT - 1, y))  
        mouse_pressed = pygame.mouse.get_pressed()[0]  

        # Capture data at intervals
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            if mouse_pressed:
                if not is_drawing:
                    # First point after clicking: flag = 0
                    doodle_data.append([x, y, 0, 0])
                else:
                    # Continuation point: flag = 1
                    doodle_data.append([x, y, 1, 0])
                is_drawing = True
            else:
                is_drawing = False  # Reset drawing state when mouse is released
            last_capture_time = current_time

        # Draw current doodle path
        if len(doodle_data) > 1:
            for i in range(1, len(doodle_data)):
                if doodle_data[i][2] == 1:  # Draw line only if the flag is 1
                    pygame.draw.line(screen, BLACK, (doodle_data[i-1][0], doodle_data[i-1][1]),
                                     (doodle_data[i][0], doodle_data[i][1]), 3)

        pygame.display.flip()

    # Mark the last point as a termination point
    if doodle_data:
        doodle_data[-1][-1] = 1

    # Append doodle data to the CSV file only if there is meaningful data
    if doodle_data:
        csv_file = "extra_classes.csv"
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([doodle_class, doodle_data])

    pygame.quit()

def main():
    print("Welcome to the Doodle Recorder!")
    print("Type the class name to draw an instance or 'quit' to exit.")
    while True:
        doodle_class = input("Enter class name (or 'quit' to exit): ").strip()
        if doodle_class.lower() == "quit":
            print("Exiting the Doodle Recorder. Goodbye!")
            break
        elif doodle_class:
            print(f"Start drawing for class '{doodle_class}'. Close the window when done.")
            record_doodle(doodle_class)
        else:
            print("Please enter a valid class name.")

if __name__ == "__main__":
    main()
