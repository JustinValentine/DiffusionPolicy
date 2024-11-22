import matplotlib.pyplot as plt
import numpy as np

def softmax(values):
    exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
    return exp_values / np.sum(exp_values)

def plot_drawing(data, output_file="drawing_plot.png"):

    x_prev, y_prev = None, None  
    plt.figure(figsize=(6, 6))
    plt.axis([0, 255, 0, 255])
    plt.gca().invert_yaxis() 

    for point in data:
        x, y, on_paper, termination = point

        softmax_values = softmax([on_paper, termination])
        print(on_paper)
        on_paper_prob = softmax_values[0]
        termination_prob = softmax_values[1]

        if on_paper_prob > 0.5:  # If termination probability > 0.5, stop drawing
            if x_prev is not None and y_prev is not None:
                plt.plot([x_prev, x], [y_prev, y], color="black")  # Draw a line

        if termination_prob > 0.5:
            break

        x_prev, y_prev = x, y


    plt.title("Generated Doodle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close() 


data = [[33.19672107696533, 218.24176043272018, -0.8204028010368347, -0.7529687285423279], [3.5357558727264404, 183.84706623852253, 0.7373279333114624, -0.7748672366142273], [124.7210746910423, 48.108716905117035, -0.8927876353263855, -0.8651196360588074], [110.46665590256453, 98.24147310107946, 0.22448015213012695, -0.766822338104248], [101.40329673886299, 120.44019685126841, 0.8515060544013977, -0.8299556374549866], [155.27717638760805, 69.44883920252323, 0.2679109573364258, -0.7845658659934998], [136.1643416248262, 67.48266890645027, 0.8566989898681641, -0.7742918133735657], [115.07193088531494, 88.68688009679317, -1.0, -0.8730380535125732], [120.4693161137402, 127.99189141311217, 0.8207942843437195, -0.8262119889259338], [101.14646852016449, 136.12466510385275, 0.8753582835197449, -0.8586341142654419], [158.26416950672865, 114.22434646636248, 0.8229447603225708, -0.8723497986793518], [147.36573681235313, 84.69174787402153, -0.16762052476406097, -0.8941418528556824], [132.65065639279783, 79.82902340590954, -0.21839262545108795, -0.8741228580474854], [153.2132601365447, 96.25944815576077, 0.9088683128356934, -0.8920520544052124], [166.26441411674023, 100.44318705797195, 0.8442657589912415, -0.8734880685806274], [200.10270416736603, 93.29102843999863, 0.8500511050224304, -0.9013375639915466], [206.98560923337936, 103.32326341420412, 0.8573815226554871, -0.8928782343864441], [162.95718558132648, 159.53246735036373, 0.8032117486000061, -0.8395030498504639], [156.09300021082163, 204.5262672007084, 0.8465934991836548, -0.1583372801542282], [30.71044608950615, 99.55941718071699, 0.3395181894302368, -0.4986545741558075], [16.197641640901566, 37.05631837248802, 0.2329060286283493, -0.5966725945472717], [14.073464423418045, 26.556303799152374, 0.5115069150924683, -0.5772231817245483], [24.168131947517395, 15.443230122327805, 0.4678264856338501, -0.591749906539917], [5.891295075416565, 28.113711029291153, 0.5075439810752869, -0.6364170908927917], [28.347079306840897, 28.084969371557236, 0.5996002554893494, -0.6989631056785583], [6.935752630233765, 28.199282437562943, 0.6387770175933838, -0.6241894364356995], [34.62487444281578, 8.412976562976837, 0.21280483901500702, -0.7246147394180298], [44.30077403783798, 42.32896864414215, 0.2604268491268158, -0.7893773913383484], [76.03999212384224, 42.33358159661293, 0.556057870388031, -0.7617852091789246], [73.02540868520737, 47.26260110735893, 0.6886929273605347, -0.7649556398391724], [97.35051970928907, 53.65571245551109, 0.644497275352478, -0.7870078086853027], [100.16906406730413, 48.55823278427124, 0.42524370551109314, -0.7805288434028625], [124.11993317306042, 96.08021177351475, 0.7120077610015869, -0.6430525779724121], [142.77509769424796, 42.67026633024216, 0.14760345220565796, -0.625199556350708], [167.38732986152172, 53.52506026625633, 0.7294389605522156, -0.7075307965278625], [126.34765625931323, 92.73853428661823, 0.6561369895935059, -0.7605843544006348], [64.54335182905197, 129.40303798997775, 0.48976951837539673, -0.30277520418167114], [24.37269017100334, 49.116506427526474, 0.46082496643066406, 0.00823196955025196], [8.52063238620758, 55.12320891022682, 0.26962336897850037, -0.06825186312198639], [9.671757817268372, 18.92030194401741, 0.04279081150889397, -0.09017565101385117], [8.420325368642807, 15.212453305721283, 0.08244600147008896, -0.055196117609739304], [10.764259994029999, 17.671232968568802, 0.02543693780899048, 0.018362190574407578], [9.373876601457596, 6.771472245454788, 0.026745818555355072, -0.047949496656656265], [5.806483626365662, 8.19546863436699, 0.010230155661702156, -0.06165149062871933], [7.657691091299057, 4.398431181907654, -0.12233374267816544, -0.08228809386491776], [5.024911165237427, 10.073008686304092, -0.010852077975869179, -0.027709651738405228], [7.908591628074646, 6.30380854010582, 0.03297584503889084, -0.04259300231933594], [3.076299726963043, 5.805655270814896, -0.012030813843011856, -0.07049091905355453], [8.164393901824951, 6.176500171422958, 0.009056603536009789, -0.052246324717998505], [5.058653354644775, 7.600086182355881, 0.020027875900268555, -0.0047928993590176105], [7.2716546058654785, 7.962974309921265, 0.02303464524447918, -0.03739672899246216], [5.602221786975861, 8.390838950872421, -0.018056681379675865, -0.03559279069304466], [7.861390560865402, 7.229218482971191, 0.023161238059401512, -0.04340201988816261], [5.387221723794937, 8.276875466108322, 0.021937236189842224, -0.03687424957752228], [7.11019366979599, 6.8402257561683655, -0.08191091567277908, -0.059565868228673935], [4.757420718669891, 7.447501569986343, 0.010920645669102669, -0.009284867905080318], [7.9335638880729675, 8.475323617458344, -0.011644686572253704, -0.046203333884477615], [5.366178452968597, 5.766912549734116, -0.003529587062075734, -0.018525585532188416], [6.886043697595596, 4.88969162106514, 0.0037842856254428625, -0.034616585820913315], [6.477542817592621, 8.880860656499863, 0.00900083314627409, -0.033021315932273865], [6.302425414323807, 5.8975571393966675, 0.01667255535721779, -0.03794264420866966], [3.64731028676033, 8.524348586797714, -0.058222588151693344, -0.023884959518909454], [5.226862728595734, 3.983318656682968, 0.009688438847661018, -0.024753352627158165], [10.19499734044075, 5.969715267419815, 0.03764183074235916, -0.056404273957014084]]


plot_drawing(data, output_file="drawing_output.png")